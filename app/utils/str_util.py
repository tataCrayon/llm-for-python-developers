class StrUtil:
    @staticmethod
    def text_analyzer(text: str):
        """
        输入：一个字符串文本
        返回：字典。字数、词数、行数
        """
        if not isinstance(text, str):
            raise TypeError("输入必须是字符串类型")
        return {"char_size": len(text), "word_size": len(text.split()), "line_size": len(text.splitlines())}

    @staticmethod
    def case_converter(text: str, case_type: str = 'upper'):
        """
        输入：字符串文本
        返回：case_type指定的大小写形式的字符串
        """
        if not isinstance(text, str):
            raise TypeError("输入必须是字符串类型")
        if case_type == 'upper':
            return text.upper()
        elif case_type == 'lower':
            return text.lower()
        return None


