@echo off
deroluna-miner -d 203.0.113.10:10100 -w DERO_WALLET_PLACEHOLDER -t 20
timeout 3
goto loop