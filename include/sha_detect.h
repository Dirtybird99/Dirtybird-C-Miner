#ifndef SHA_DETECT_H
#define SHA_DETECT_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#if defined(_WIN32)

#include <windows.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

static inline int has_sha_ni_support(void)
{
#if defined(__x86_64__)
  int supported = 0;

  HANDLE read_pipe, write_pipe;
  SECURITY_ATTRIBUTES sa = {sizeof(sa), NULL, TRUE};
  if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) return 0;

  char exe_path[MAX_PATH];
  if (!GetModuleFileNameA(NULL, exe_path, MAX_PATH)) return 0;

  STARTUPINFOA si = {sizeof(si)};
  PROCESS_INFORMATION pi;
  memset(&pi, 0, sizeof(pi));

  si.dwFlags |= STARTF_USESTDHANDLES;
  si.hStdOutput = write_pipe;
  si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
  si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);

  // Set environment variable to tell child process to run SHA-NI test
  SetEnvironmentVariableA("__SHA_NI_PROBE_CHILD__", "1");

  if (!CreateProcessA(NULL, exe_path, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
    SetEnvironmentVariableA("__SHA_NI_PROBE_CHILD__", NULL);
    CloseHandle(read_pipe);
    CloseHandle(write_pipe);
    return 0;
  }

  // Clear environment variable in parent
  SetEnvironmentVariableA("__SHA_NI_PROBE_CHILD__", NULL);

  CloseHandle(write_pipe);
  WaitForSingleObject(pi.hProcess, INFINITE);

  char buf[1] = {0};
  DWORD read = 0;
  if (ReadFile(read_pipe, buf, 1, &read, NULL) && read == 1 && buf[0] == '1')
    supported = 1;

  CloseHandle(read_pipe);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  return supported;
#else
  return 0; // Not x86_64
#endif
}

// DERO Miner: Removed constructor-based SHA-NI detection.
// The original code used GetConsoleMode to detect if running as child process,
// but this fails in MSYS2/mintty terminals (not real Windows consoles),
// causing the main process to exit immediately with "1".
//
// SHA-NI detection is now done only when has_sha_ni_support() is called,
// using an environment variable to distinguish parent from child.
#ifdef __x86_64__
static void sha_probe_child_check(void)
{
  // Only run SHA-NI test if we're the child process
  const char* probe_marker = getenv("__SHA_NI_PROBE_CHILD__");
  if (probe_marker == NULL || strcmp(probe_marker, "1") != 0) {
    return; // Not the child process, return normally
  }

  // We're the child process - test SHA-NI and report result
  HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hStdout == INVALID_HANDLE_VALUE) ExitProcess(1);

  __m128i a = _mm_setzero_si128();
  __m128i b = _mm_setzero_si128();
  __m128i c = _mm_setzero_si128();
  __m128i r = _mm_sha256rnds2_epu32(a, b, c);
  volatile uint32_t dummy = _mm_extract_epi32(r, 0);
  (void)dummy;
  DWORD written;
  WriteFile(hStdout, "1", 1, &written, NULL);
  ExitProcess(0);
}

// Call this early in main() or as a constructor - it only exits if we're the child
__attribute__((constructor)) static void sha_probe_child_windows(void)
{
  sha_probe_child_check();
}
#endif

#elif defined(__unix__) || defined(__APPLE__)

#include <unistd.h>
#include <sys/wait.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

static inline int has_sha_ni_support(void)
{
#if defined(__x86_64__)
  int pipefd[2];
  if (pipe(pipefd) != 0) return 0;

  pid_t pid = fork();
  if (pid < 0) return 0;

  if (pid == 0) {
    close(pipefd[0]);

    __m128i a = _mm_setzero_si128();
    __m128i b = _mm_setzero_si128();
    __m128i c = _mm_setzero_si128();
    __m128i r = _mm_sha256rnds2_epu32(a, b, c);
    volatile uint32_t dummy = _mm_extract_epi32(r, 0);
    (void)dummy;

    write(pipefd[1], "1", 1);
    _exit(0);
  } else {
    close(pipefd[1]);
    char result = 0;
    read(pipefd[0], &result, 1);
    close(pipefd[0]);
    waitpid(pid, NULL, 0);
    return result == '1';
  }
#else
  return 0; // Not x86_64
#endif
}

#else

static inline int has_sha_ni_support(void) { return 0; }

#endif

// Cached version
static inline int has_sha_ni_support_cached(void)
{
  static int cached = -1;
  if (cached == -1)
    cached = has_sha_ni_support();
  return cached;
}

#endif // SHA_DETECT_H
