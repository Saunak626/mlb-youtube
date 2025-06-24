class Meter(object):
    """
    这是一个抽象的基类 (Interface)，用于定义所有评估指标工具 (Meter) 的基本结构。
    所有具体的评估工具类 (如 APMeter) 都应继承此类并实现其方法。
    这确保了不同评估指标有统一的调用方式。
    """
    def reset(self):
        """
        重置/清空评估器的内部状态。
        """
        pass

    def add(self):
        """
        向评估器中添加新的数据点或一个批次的数据。
        """
        pass

    def value(self):
        """
        计算并返回当前评估指标的值。
        """
        pass
