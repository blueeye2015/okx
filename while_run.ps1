# 设置循环条件
$keepRunning = $true

while ($keepRunning) {
    # 执行 Python 脚本
    D:/ProgramData/anaconda3/envs/akshare/python.exe d:/OKex-API/getpos.py

    # 可选：在每次执行后暂停
    Start-Sleep -Seconds 100  # 暂停10秒

    # 可选：根据某种条件停止循环
    # 例如：如果某个文件存在，则停止
    #if (Test-Path "C:\path\to\stop.txt") {
    #    $keepRunning = $false
    #}
}
