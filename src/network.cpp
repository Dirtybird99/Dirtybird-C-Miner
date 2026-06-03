/*
 * network.cpp -- raw TLS WebSocket client for DERO daemon
 *
 * No Boost. No nlohmann. No frameworks. Just sockets, OpenSSL, and grit.
 * Protocol: WSS to /ws/{wallet}, receive jobs as JSON, submit shares as JSON.
 */

#include "dluna.h"
#include "hex.h"

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/rand.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <ctime>
#include <chrono>
#include <string>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <mstcpip.h>   /* struct tcp_keepalive, SIO_KEEPALIVE_VALS */
  #pragma comment(lib, "ws2_32.lib")
  typedef SOCKET sock_t;
  #define SOCK_INVALID INVALID_SOCKET
  #define sock_close closesocket
  #define sock_errno WSAGetLastError()
#else
  #include <unistd.h>
  #include <sys/socket.h>
  #include <sys/types.h>
  #include <netinet/tcp.h>
  #include <netdb.h>
  #include <arpa/inet.h>
  typedef int sock_t;
  #define SOCK_INVALID (-1)
  #define sock_close close
  #define sock_errno errno
#endif

extern MinerState G;

/* ---------- globals ---------- */

static SSL_CTX *g_ctx;
static SSL     *g_ssl;
static sock_t   g_sock = SOCK_INVALID;

/* ---------- minimal JSON extractors ---------- */

/* Extract a string value for a given key from raw JSON.
 * Handles: "key":"value" -- no nested objects, no escapes.
 * Returns empty string on miss. */
static std::string json_str(const char *json, const char *key)
{
	char pat[128];
	int n = snprintf(pat, sizeof pat, "\"%s\":\"", key);
	if (n <= 0) return {};

	const char *p = strstr(json, pat);
	if (!p) return {};
	p += n;

	const char *q = strchr(p, '"');
	if (!q) return {};
	return std::string(p, q - p);
}

/* Extract an int64 value for a given key from raw JSON.
 * Handles: "key":12345 -- bare integer, no quotes.
 * Returns 0 on miss. */
static int64_t json_int(const char *json, const char *key)
{
	char pat[128];
	int n = snprintf(pat, sizeof pat, "\"%s\":", key);
	if (n <= 0) return 0;

	const char *p = strstr(json, pat);
	if (!p) return 0;
	p += n;

	/* skip whitespace */
	while (*p == ' ' || *p == '\t') p++;
	if (*p == '"') return 0; /* it's a string, not an int */

	return strtoll(p, nullptr, 10);
}

/* ---------- base64 for Sec-WebSocket-Key ---------- */

static const char b64tab[] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const uint8_t *d, int len)
{
	std::string out;
	out.reserve(((len + 2) / 3) * 4);
	for (int i = 0; i < len; i += 3) {
		uint32_t v = (uint32_t)d[i] << 16;
		if (i + 1 < len) v |= (uint32_t)d[i+1] << 8;
		if (i + 2 < len) v |= d[i+2];
		out += b64tab[(v >> 18) & 0x3f];
		out += b64tab[(v >> 12) & 0x3f];
		out += (i + 1 < len) ? b64tab[(v >> 6) & 0x3f] : '=';
		out += (i + 2 < len) ? b64tab[v & 0x3f] : '=';
	}
	return out;
}

/* ---------- socket helpers ---------- */

static void set_tcp_nodelay(sock_t s)
{
	int on = 1;
	setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&on, sizeof on);
}

/* Enable TCP keepalive tuned for fast dead-link detection -- lets the OS surface
 * a black-holed socket (NAT/conntrack drop, router reboot, cable pull -- no
 * RST/FIN) as an error.
 *
 * POSIX: keepalive exhaustion => ETIMEDOUT, which is NOT in ssl_read_is_timeout's
 * keep-set, so recv_all returns -1 and we reconnect. Works directly (e.g. HiveOS
 * rigs over a remote pool).
 *
 * Windows: keepalive exhaustion surfaces as WSAETIMEDOUT (10060) -- the SAME code
 * ssl_read_is_timeout deliberately KEEPs (dual-purpose with the 50ms poll) -- so
 * it does NOT flip recv_all to dead. Here keepalive is only OS-side resource
 * release / NAT refresh; a silently-dead link on Windows is a documented gap (see
 * the deferred liveness-watchdog follow-up). We do NOT special-case 10060 -- that
 * would regress the verified read fix. */
static void set_keepalive(sock_t s)
{
#ifdef _WIN32
	struct tcp_keepalive ka;
	ka.onoff             = 1;
	ka.keepalivetime     = 10000; /* ms idle before first probe */
	ka.keepaliveinterval = 1000;  /* ms between probes */
	DWORD bytes = 0;
	WSAIoctl(s, SIO_KEEPALIVE_VALS, &ka, sizeof ka,
	         nullptr, 0, &bytes, nullptr, nullptr);
#else
	int on = 1;
	setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof on);
	int idle = 10, intvl = 1, cnt = 10;
#ifdef TCP_KEEPIDLE
	setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE,  &idle,  sizeof idle);
#endif
#ifdef TCP_KEEPINTVL
	setsockopt(s, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof intvl);
#endif
#ifdef TCP_KEEPCNT
	setsockopt(s, IPPROTO_TCP, TCP_KEEPCNT,   &cnt,   sizeof cnt);
#endif
#endif
}

static void set_timeout_ms(sock_t s, int ms)
{
#ifdef _WIN32
	DWORD tv = ms;
	setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof tv);
#else
	struct timeval tv;
	tv.tv_sec = ms / 1000;
	tv.tv_usec = (ms % 1000) * 1000;
	setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
}

/* Send-side timeout. Set ONCE per connection (generous) and never toggled --
 * keeping it out of set_timeout_ms() is deliberate: that helper drops RCVTIMEO
 * to 50ms for the mining loop, and a 50ms SO_SNDTIMEO would make submit_share()
 * /pong sends spuriously fail under transient TCP backpressure and force a
 * needless reconnect. SNDTIMEO only needs to bound a wedged handshake/send. */
static void set_snd_timeout_ms(sock_t s, int ms)
{
#ifdef _WIN32
	DWORD tv = ms;
	setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof tv);
#else
	struct timeval tv;
	tv.tv_sec = ms / 1000;
	tv.tv_usec = (ms % 1000) * 1000;
	setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
}

static int send_all(const void *buf, int len)
{
	const char *p = (const char *)buf;
	int sent = 0;
	while (sent < len) {
		int n = SSL_write(g_ssl, p + sent, len - sent);
		if (n <= 0) {
			int se = sock_errno;
			int sslerr = SSL_get_error(g_ssl, n);
			log_line("WARN", "SSL_write failed (ssl_err=%d sock_err=%d)", sslerr, se);
			return -1;
		}
		sent += n;
	}
	return sent;
}

/* Classify a non-positive SSL_read result. A socket read timeout (SO_RCVTIMEO)
 * is a recoverable "no data yet"; everything else means the connection is gone
 * and we must reconnect.
 *
 * On a Windows blocking socket a read timeout surfaces as SSL_ERROR_SYSCALL +
 * WSAETIMEDOUT (10060) -- NOT SSL_ERROR_WANT_READ -- so the keep-set is an
 * explicit allowlist. Anything unrecognised defaults to "dead" -> reconnect
 * (backoff-bounded), never to "keep" (which wedges the miner spinning shares
 * into a dead socket). Notably OpenSSL 3.x reports a peer close without
 * close_notify as SSL_ERROR_SSL (1) / sock_err 0, which this correctly treats
 * as a dead connection. */
static bool ssl_read_is_timeout(int sslerr, int se)
{
	if (sslerr == SSL_ERROR_WANT_READ || sslerr == SSL_ERROR_WANT_WRITE)
		return true;
	if (sslerr == SSL_ERROR_SYSCALL) {
#ifdef _WIN32
		return se == WSAETIMEDOUT || se == WSAEWOULDBLOCK;
#else
		return se == EAGAIN || se == EWOULDBLOCK || se == EINTR;
#endif
	}
	return false;
}

/* Read exactly len bytes. Returns len on success, -1 on a dead connection
 * (caller must reconnect), and -2 ONLY when first==true and the very first
 * read times out (no message pending). Mid-frame timeouts loop and wait for
 * the rest of the frame, so a payload landing a few ms after its header never
 * triggers a spurious reconnect. */
static int recv_all(void *buf, int len, bool first)
{
	char *p = (char *)buf;
	int got = 0;
	int stall_polls = 0;
	/* Bound a stalled mid-frame read: a peer that sends a frame header then
	 * goes silent (or vanishes with no RST/FIN) must not wedge the thread. */
	const int MAX_STALL_POLLS = 200; /* 200 * 50ms SO_RCVTIMEO = 10s */
	while (got < len) {
		int n = SSL_read(g_ssl, p + got, len - got);
		if (n > 0) { got += n; stall_polls = 0; continue; }
		if (n == 0) return -1;          /* clean EOF -- peer closed, always dead */

		int se = sock_errno;
		int sslerr = SSL_get_error(g_ssl, n);
		if (ssl_read_is_timeout(sslerr, se)) {
			if (first && got == 0)
				return -2;          /* no message pending */
			if (G.quit.load())
				return -1;          /* don't block shutdown mid-frame */
			if (++stall_polls >= MAX_STALL_POLLS) {
				log_line("WARN", "stalled mid-frame %d ms — reconnecting",
				         MAX_STALL_POLLS * 50);
				return -1;          /* header arrived, body never did */
			}
			continue;                   /* mid-frame: wait for the rest */
		}
		log_line("WARN", "connection error (ssl_err=%d sock_err=%d) — reconnecting",
		         sslerr, se);
		return -1;                          /* dead connection */
	}
	return got;
}

/* ---------- SSL init ---------- */

static bool ssl_init(void)
{
	if (g_ctx) return true;

	SSL_library_init();
	SSL_load_error_strings();
	OpenSSL_add_all_algorithms();

	g_ctx = SSL_CTX_new(TLS_client_method());
	if (!g_ctx) return false;

	/* Mining daemons use self-signed certs. Don't verify. */
	SSL_CTX_set_verify(g_ctx, SSL_VERIFY_NONE, nullptr);
	return true;
}

/* ---------- WebSocket connect ---------- */

/* Per-read SO_RCVTIMEO/SO_SNDTIMEO during the connect+upgrade phase. Generous
 * enough for a real TLS handshake over a high-RTT link; bounds a stalled peer. */
static const int CONNECT_TIMEOUT_MS = 10000;
/* Wall-clock ceiling for the whole TCP+TLS+HTTP-upgrade phase. A peer can defeat
 * the per-read timeout by trickling one byte every few seconds (each SSL_read
 * returns >0, so the per-read timer never fires and \r\n\r\n never arrives);
 * this deadline bounds that. Headroom over CONNECT_TIMEOUT_MS so a single slow
 * (but legitimate) read doesn't trip it. */
static const long long CONNECT_DEADLINE_MS = 20000;

static void cleanup(void)
{
	if (g_ssl) { SSL_shutdown(g_ssl); SSL_free(g_ssl); g_ssl = nullptr; }
	if (g_sock != SOCK_INVALID) { sock_close(g_sock); g_sock = SOCK_INVALID; }
	G.connected.store(false);
}

static bool ws_connect(void)
{
	cleanup();

	log_line("INFO", "Connecting (%s:%u)", G.host.c_str(), G.port);
	auto connect_start = std::chrono::steady_clock::now();

	/* DNS resolve */
	struct addrinfo hints{}, *res = nullptr;
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;

	char portstr[16];
	snprintf(portstr, sizeof portstr, "%u", G.port);

	if (getaddrinfo(G.host.c_str(), portstr, &hints, &res) != 0 || !res) {
		log_line("ERROR", "DNS resolve failed for %s", G.host.c_str());
		return false;
	}

	/* TCP connect */
	g_sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
	if (g_sock == SOCK_INVALID) {
		freeaddrinfo(res);
		return false;
	}

	if (connect(g_sock, res->ai_addr, (int)res->ai_addrlen) != 0) {
		log_line("ERROR", "TCP connect failed to %s:%u",
			G.host.c_str(), G.port);
		freeaddrinfo(res);
		sock_close(g_sock);
		g_sock = SOCK_INVALID;
		return false;
	}
	freeaddrinfo(res);

	set_tcp_nodelay(g_sock);
	set_keepalive(g_sock);

	/* Bound the handshake + HTTP-upgrade phase. The socket is blocking, so
	 * without this SSL_connect() and the upgrade SSL_read loop would wait
	 * forever on a peer that finishes TCP but stalls during TLS or never sends
	 * the response terminator. 10s is generous for a real handshake over a
	 * high-RTT link yet bounds a wedge. This RCVTIMEO is replaced by the 50ms
	 * mining-loop timeout (set_timeout_ms below) once the upgrade succeeds;
	 * SNDTIMEO is set once and kept for the connection's whole life. */
	set_timeout_ms(g_sock, CONNECT_TIMEOUT_MS);
	set_snd_timeout_ms(g_sock, CONNECT_TIMEOUT_MS);

	/* TLS handshake */
	g_ssl = SSL_new(g_ctx);
	SSL_set_fd(g_ssl, (int)g_sock);
	if (SSL_connect(g_ssl) != 1) {
		log_line("ERROR", "TLS handshake failed");
		cleanup();
		return false;
	}

	/* HTTP upgrade to WebSocket */
	uint8_t keyraw[16];
	RAND_bytes(keyraw, 16);
	std::string wskey = base64_encode(keyraw, 16);

	char req[1024];
	int reqlen = snprintf(req, sizeof req,
		"GET /ws/%s HTTP/1.1\r\n"
		"Host: %s:%u\r\n"
		"Upgrade: websocket\r\n"
		"Connection: Upgrade\r\n"
		"Sec-WebSocket-Key: %s\r\n"
		"Sec-WebSocket-Version: 13\r\n"
		"\r\n",
		G.wallet.c_str(), G.host.c_str(), G.port, wskey.c_str());

	if (send_all(req, reqlen) < 0) {
		log_line("ERROR", "failed to send HTTP upgrade");
		cleanup();
		return false;
	}

	/* Read HTTP response -- just need "101".
	 * Guards: G.quit lets Ctrl-C escape a stalled upgrade between reads; the
	 * wall-clock deadline bounds a peer that trickles bytes forever without
	 * ever sending the \r\n\r\n terminator (the per-read CONNECT_TIMEOUT_MS
	 * alone can't catch that -- each dribbled byte makes SSL_read return >0). */
	char resp[2048];
	int total = 0;
	for (;;) {
		if (G.quit.load()) { cleanup(); return false; }
		long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - connect_start).count();
		if (elapsed >= CONNECT_DEADLINE_MS) {
			log_line("ERROR", "WebSocket upgrade stalled (%lld ms)", elapsed);
			cleanup();
			return false;
		}
		int n = SSL_read(g_ssl, resp + total, (int)sizeof(resp) - total - 1);
		if (n <= 0) { cleanup(); return false; }
		total += n;
		resp[total] = '\0';
		if (strstr(resp, "\r\n\r\n")) break;
		if (total >= (int)sizeof(resp) - 1) { cleanup(); return false; }
	}

	if (!strstr(resp, " 101 ")) {
		log_line("ERROR", "WebSocket upgrade rejected");
		cleanup();
		return false;
	}

	set_timeout_ms(g_sock, 50);
	G.connected.store(true);
	long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - connect_start).count();
	log_line("INFO", "Connected (%s:%u) (%lld ms)", G.host.c_str(), G.port, ms);
	return true;
}

/* ---------- WebSocket framing ---------- */

static int ws_send_message(const void *data, int len, int opcode)
{
	uint8_t frame[14]; /* max header: 2 + 8 + 4 */
	int hdrlen = 0;

	frame[0] = 0x80 | (opcode & 0x0f); /* FIN + opcode */

	/* Client frames MUST be masked */
	if (len < 126) {
		frame[1] = 0x80 | (uint8_t)len;
		hdrlen = 2;
	} else if (len < 65536) {
		frame[1] = 0x80 | 126;
		frame[2] = (len >> 8) & 0xff;
		frame[3] = len & 0xff;
		hdrlen = 4;
	} else {
		frame[1] = 0x80 | 127;
		uint64_t l = (uint64_t)len;
		for (int i = 0; i < 8; i++)
			frame[2 + i] = (l >> (56 - 8*i)) & 0xff;
		hdrlen = 10;
	}

	/* Masking key */
	uint8_t mask[4];
	RAND_bytes(mask, 4);
	memcpy(frame + hdrlen, mask, 4);
	hdrlen += 4;

	if (send_all(frame, hdrlen) < 0) return -1;

	/* Mask and send payload */
	if (len > 0) {
		/* Copy to avoid mutating caller's data */
		std::string masked((const char *)data, len);
		for (int i = 0; i < len; i++)
			masked[i] ^= mask[i & 3];
		if (send_all(masked.data(), len) < 0) return -1;
	}
	return 0;
}

/* Convenience: send text frame */
static int ws_send_text(const std::string &msg)
{
	return ws_send_message(msg.data(), (int)msg.size(), 0x1);
}

/* Read one complete WebSocket message. Returns payload length, -1 on error, 0 on timeout. */
static int ws_read_message(std::string &out)
{
	out.clear();
	uint8_t hdr[2];
	int n = recv_all(hdr, 2, true);
	if (n == -2) return 0; /* timeout on first read — no data available */
	if (n < 0) return -1;  /* real error */

	bool fin = (hdr[0] & 0x80) != 0;
	int opcode = hdr[0] & 0x0f;
	bool masked = (hdr[1] & 0x80) != 0;
	uint64_t plen = hdr[1] & 0x7f;

	if (plen == 126) {
		uint8_t ext[2];
		if (recv_all(ext, 2, false) < 0) return -1;
		plen = ((uint64_t)ext[0] << 8) | ext[1];
	} else if (plen == 127) {
		uint8_t ext[8];
		if (recv_all(ext, 8, false) < 0) return -1;
		plen = 0;
		for (int i = 0; i < 8; i++)
			plen = (plen << 8) | ext[i];
	}

	if (plen > 1024 * 1024) return -1;
	/* RFC6455 5.5: control frames (opcode >= 0x8) carry <=125-byte payloads and
	 * must not be fragmented. plen is fully decoded above, so this also rejects a
	 * control frame that illegally used the 126/127 extended-length encoding. */
	if (opcode >= 0x8 && plen > 125) return -1;

	uint8_t mask[4] = {};
	if (masked) {
		if (recv_all(mask, 4, false) < 0) return -1;
	}

	if (plen > 0) {
		out.resize((size_t)plen);
		if (recv_all(&out[0], (int)plen, false) < 0) return -1;
		if (masked) {
			for (uint64_t i = 0; i < plen; i++)
				out[i] ^= mask[i & 3];
		}
	}

	/* Control frames */
	if (opcode == 0x9) { /* ping -> pong */
		/* Capture the send result before clearing out -- out.data() is the pong
		 * payload. A failed pong send means the socket is dead, so surface it now
		 * (-1 -> reconnect) instead of masking the death for one more read cycle. */
		int sr = ws_send_message(out.data(), (int)out.size(), 0xA);
		out.clear();
		if (sr < 0) return -1;
		return 0;
	}
	if (opcode == 0x8) return -1; /* close */
	if (!fin) return -1;          /* fragmented, bail */

	return (int)plen;
}

/* ---------- job handling ---------- */

/* Returns true iff a well-formed job was parsed (passed the blob/jobid/size
 * guards). The network loop ORs this into a per-session got_job flag that gates
 * the reconnect-backoff reset: a daemon error-only or malformed frame returns
 * false and does NOT arm a fast reconnect. */
static bool handle_job(const std::string &json)
{
	std::string blob = json_str(json.c_str(), "blockhashing_blob");
	std::string jid  = json_str(json.c_str(), "jobid");
	int64_t h        = json_int(json.c_str(), "height");
	int64_t diff     = json_int(json.c_str(), "difficultyuint64");
	int64_t miniblks = json_int(json.c_str(), "miniblocks");
	int64_t blks     = json_int(json.c_str(), "blocks");
	int64_t rej      = json_int(json.c_str(), "rejected");
	std::string err  = json_str(json.c_str(), "lasterror");

	if (!err.empty())
		log_line("WARN", "daemon error: %s", err.c_str());

	if (blob.empty() || jid.empty()) return false;
	if (blob.size() != MINIBLOCK_SIZE * 2) return false;

	bool workChanged = false;

	{
		std::lock_guard<std::mutex> lk(G.jobMutex);
		workChanged =
			blob != G.blob ||
			jid  != G.jobId ||
			h    != G.height ||
			(uint64_t)diff != G.difficulty.load(std::memory_order_relaxed);

		G.blob = blob;
		hexstrToBytes(blob, G.blobBin);
		G.jobId = jid;
		G.height = h;
		G.difficulty.store((uint64_t)diff);
		G.accepted.store(miniblks, std::memory_order_relaxed);
		G.blocks.store(blks, std::memory_order_relaxed);
		G.rejected.store(rej, std::memory_order_relaxed);

		if (workChanged)
			G.jobEpoch.fetch_add(1);
	}

	if (workChanged) {
		G.newJob.notify_all();
		if (g_verbose)
			log_line("INFO", "job %s  height=%lld  diff=%lld",
				jid.c_str(), (long long)h, (long long)diff);
	}
	return true;
}

/* ---------- share submission ---------- */

/* Returns false if the send failed -- the connection is dead and the caller
 * must tear it down and reconnect. The dropped share is fine to lose: it would
 * be stale on the new connection anyway. */
static bool submit_share(void)
{
	std::string jid, blob;
	uint64_t epoch;

	{
		std::lock_guard<std::mutex> lk(G.submitMutex);
		if (!G.submitReady.load()) return true;
		jid   = G.submitJobId;
		blob  = G.submitBlob;
		epoch = G.submitEpoch;
		G.submitReady.store(false);
	}

	/* Stale check */
	if (epoch != G.jobEpoch.load()) {
		G.staleDrops.fetch_add(1);
		if (g_verbose)
			log_line("INFO", "stale share dropped (epoch %llu vs %llu)",
				(unsigned long long)epoch, (unsigned long long)G.jobEpoch.load());
		return true;
	}

	/* Build JSON manually -- no allocator overhead */
	char json[512];
	snprintf(json, sizeof json,
		"{\"jobid\":\"%s\",\"mbl_blob\":\"%s\"}",
		jid.c_str(), blob.c_str());

	if (ws_send_text(json) == 0) {
		G.submitted.fetch_add(1);
		if (g_verbose)
			log_line("INFO", "share submitted for job %s", jid.c_str());
		return true;
	}
	G.sendFails.fetch_add(1);
	log_line("WARN", "share send failed — reconnecting");
	return false;
}

/* Sleep up to ms, but in <=250ms slices that re-check G.quit, so a Ctrl-C during
 * a grown reconnect backoff (up to 30s) doesn't lag net.join(). This only chunks
 * the wait at the call site; dluna_sleep_ms's global contract is unchanged. */
static void backoff_sleep(int ms)
{
	const int SLICE_MS = 250;
	while (ms > 0 && !G.quit.load()) {
		int slice = ms < SLICE_MS ? ms : SLICE_MS;
		dluna_sleep_ms(slice);
		ms -= slice;
	}
}

/* ---------- network thread ---------- */

void network_thread(void)
{
#ifdef _WIN32
	WSADATA wsa;
	WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

	if (!ssl_init()) {
		log_line("ERROR", "SSL init failed");
		return;
	}

	int backoff_ms = 1000;
	const int max_backoff = 30000;
	/* A session only earns a fast (1s) reconnect if it proved useful -- delivered
	 * >=1 job or stayed up MIN_UPTIME_MS. A pool that accepts the WS upgrade then
	 * instantly drops/throttles us (rabidmining's rapid-reconnect throttle)
	 * satisfies neither, so backoff grows instead of pinning at the 1s floor and
	 * hammering the throttle. */
	const int MIN_UPTIME_MS = 10000;

	while (!G.quit.load()) {
		if (!ws_connect()) {
			if (G.quit.load()) break; /* don't sleep into a backoff on shutdown */
			log_line("WARN", "reconnect in %d ms", backoff_ms);
			backoff_sleep(backoff_ms);
			backoff_ms = std::min(backoff_ms * 2, max_backoff);
			continue;
		}
		/* DO NOT reset backoff here -- ws_connect() succeeding (TCP+TLS+101) only
		 * proves the pool accepted us, not that the session is useful. The reset
		 * is gated below on got_job / uptime. */
		bool got_job = false;
		auto session_start = std::chrono::steady_clock::now();

		/* Main loop: interleave recv and submit.
		 * SO_RCVTIMEO is 50ms so we don't block long. */
		while (!G.quit.load()) {
			/* Try to read a message */
			std::string msg;
			int rc = ws_read_message(msg);

			if (rc < 0) {
				log_line("WARN", "connection lost");
				break; /* reconnect */
			}
			if (rc > 0 && !msg.empty()) {
				if (handle_job(msg))
					got_job = true; /* this session is now "useful" */
			}

			/* Check for pending share submission. A send failure means the
			 * socket is dead -- break out to reconnect (recv usually catches
			 * the drop first, but this is the belt-and-braces path). */
			if (G.submitReady.load() && !submit_share())
				break;
		}

		cleanup();
		if (!G.quit.load()) {
			/* Backoff-gate: only a session that delivered a job or survived
			 * MIN_UPTIME_MS earns a fast 1s reconnect; otherwise the pool
			 * accepted then dropped/throttled us -- let backoff grow. */
			long long uptime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - session_start).count();
			if (got_job || uptime_ms >= MIN_UPTIME_MS)
				backoff_ms = 1000;
			log_line("WARN", "reconnect in %d ms", backoff_ms);
			backoff_sleep(backoff_ms);
			backoff_ms = std::min(backoff_ms * 2, max_backoff);
		}
	}

	cleanup();
	if (g_ctx) { SSL_CTX_free(g_ctx); g_ctx = nullptr; }

#ifdef _WIN32
	WSACleanup();
#endif
}
