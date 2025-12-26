# 国内免费GPU服务器平台及相关资源整理
## 一、主流免费GPU平台
|平台名称|网址|使用教程|特点|
| ---- | ---- | ---- | ---- |
|九天·毕昇平台|https://jiutian.10086.cn/|https://blog.csdn.net/qq1198768105/article/details/126637157|中国移动推出的AI算力平台，注册即送1000-3000算力豆，可免费使用V100显卡（显存32GB）、支持Jupyter/VSCode开发环境，终端有root权限，适合深度学习训练；算力豆有效期较短，需注意资源消耗，维护较频繁|
|阿里天池实验室|https://tianchi.aliyun.com/|https://blog.csdn.net/m0_75079597/article/details/138856414|阿里云旗下平台，提供60小时免费GPU时长（V100/P100/T4显卡），支持Notebook在线调试，适合短期训练和竞赛；单次最长使用8小时，需合理分配时间|
|百度AI Studio|https://aistudio.baidu.com/|https://ai.baidu.com/ai-doc/AISTUDIO/nk39v9kec|每周提供数十小时免费GPU算力（Tesla V100），适合PaddlePaddle框架用户，内置丰富数据集和教程；仅支持百度自研框架，无root权限，安装其他框架困难|
|OpenI启智社区|https://openi.pcl.ac.cn/|https://blog.csdn.net/weixin_44021274/article/details/131257435|集成代码管理与算力资源，提供免费GPU实例（显存16GB+），支持多种深度学习框架，适合学术研究|
|腾讯云高性能工作空间（Cloud Studio）|https://cloudstudio.net/|https://blog.csdn.net/qq_45349888/article/details/144878428|每月5万分钟免费时长，配置为T4显卡（16G显存）+8核CPU+32G内存，支持在线VSCode开发，适合AI应用部署|

## 二、云厂商免费试用资源
|平台名称|网址|使用教程|特点|
| ---- | ---- | ---- | ---- |
|华为云ModelArts|无明确网址（可通过华为云官网搜索进入）|https://www.cnblogs.com/minskiter/p/14738742.html|提供NVIDIA Tesla V100免费实例，支持TensorFlow/PyTorch等框架，适合AI模型训练与部署；需完成企业认证，试用期资源有限|
|阿里云PAI-DSW|无明确网址（可通过阿里云官网搜索进入）|https://zhuanlan.zhihu.com/p/652517173|三个月免费试用期，支持JupyterLab环境，集成主流框架（如PyTorch、TensorFlow）；需关闭实例以节省资源，显存通常为16GB|
|腾讯云GPU服务器试用|无明确网址（可通过腾讯云官网搜索进入）|https://blog.csdn.net/qq_45349888/article/details/144878428|新用户可申请免费试用GPU实例（如Tesla T4），适合短期项目测试|

## 三、国外免费GPU平台
|平台名称|网址|使用教程|特点|
| ---- | ---- | ---- | ---- |
|Kaggle|无明确网址（可通过Kaggle官网进入）|https://blog.csdn.net/weixin_42426841/article/details/143591586|Kaggle Notebook的CPU内存升级至29GB，每周免费30小时GPU时长，单次训练最长12小时，超时自动终止；需要科学上网|
|Google Colab|无明确网址（可通过Google Colab官网进入）|https://blog.csdn.net/Thebest_jack/article/details/124546083|动态分配NVIDIA T4、P100、V100等显卡，显存通常为15-16GB，支持CUDA加速，免费用户单次最长使用12小时，空闲超时30分钟自动断开；升级Pro版可延长至24小时；需要科学上网|