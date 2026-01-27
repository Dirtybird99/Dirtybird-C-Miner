@echo off
deroluna-miner -d dero-node-ch4k1pu.mysrv.cloud:10300 -w DERO_WALLET_PLACEHOLDER -t 20
timeout 3
goto loop
