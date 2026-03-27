@echo off
echo ====================================================================
echo DPProx 固定参数运行
echo ====================================================================
echo 参数配置:
echo   mu = 0.01
echo   clip_norm = 0.2
echo   noise_multiplier = 0.5
echo   epsilon = 4.0
echo   delta = 1e-5
echo ====================================================================

cd c:\Users\Gimonster\Documents\GitHub\PFLlib\system

python main.py ^
    -data MNIST ^
    -algo DPProx ^
    -model CNN ^
    -gr 10 ^
    -nc 20 ^
    -lbs 10 ^
    -lr 0.005 ^
    -ls 1 ^
    -jr 1.0 ^
    -eg 1 ^
    -mu 0.01 ^
    -dp_clip 0.2 ^
    -dp_noise 0.5 ^
    -dp_eps 4.0 ^
    -dp_delta 1e-5 ^
    -dev cuda ^
    -did 0

echo ====================================================================
echo 训练完成
echo ====================================================================
pause
