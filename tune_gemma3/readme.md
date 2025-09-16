# README

本项目用于提交 `run_tune_data.sh` 脚本到集群进行数据处理。

## 使用方法

通过 `qsub` 命令提交任务：

```bash
qsub run_tune_data.sh
```

## 参数说明

- `plk_path`：数据目录路径，脚本会在该目录下进行相关操作。