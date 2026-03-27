@echo off
echo ====================================================================
echo DPProx vs DP-Native-FedProx 对比实验
echo ====================================================================
echo.
echo 实验配置:
echo   数据集: MNIST
echo   客户端数: 20
echo   训练轮数: 10
echo   隐私预算目标: epsilon ~ 1.94
echo ====================================================================
echo.
echo [实验1] DPProx - 基准算法
echo   参数: mu=0.01, clip_norm=0.2, noise_multiplier=0.5
echo   预期隐私预算: epsilon ~ 1.94
echo ====================================================================

cd c:\Users\Gimonster\Documents\GitHub\PFLlib\system

echo.
echo ==================== DPProx 训练开始 ====================
echo.

python main.py ^
    -data MNIST ^
    -algo DPProx ^
    -m CNN ^
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

echo.
echo ====================================================================
echo [实验2] DP-Native-FedProx - 原生差分隐私算法
echo   参数: mu=0.01, clip_norm=0.2, supplementary_noise_scale=1.0
echo   预期隐私预算: epsilon ~ 1.94
echo ====================================================================
echo.
echo ==================== DP-Native-FedProx 训练开始 ====================
echo.

python main.py ^
    -data MNIST ^
    -algo DPProxNative ^
    -m CNN ^
    -gr 10 ^
    -nc 20 ^
    -lbs 10 ^
    -lr 0.005 ^
    -ls 1 ^
    -jr 1.0 ^
    -eg 1 ^
    -mu 0.01 ^
    -dp_clip 0.2 ^
    -sup_noise ^
    -sup_noise_scale 1.0 ^
    -lambda_priv 0.05 ^
    -dp_delta 1e-5 ^
    -dev cuda ^
    -did 0

echo.
echo ====================================================================
echo 对比实验完成
echo ====================================================================
pause
