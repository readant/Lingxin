"""
第1阶段：Python基础回顾

本脚本帮助您回顾Python的基础语法和常用操作。
"""

def section_1_variables():
    """1.1 变量和数据类型"""
    print("\n" + "=" * 50)
    print("1.1 变量和数据类型")
    print("=" * 50)
    
    # 基本数据类型
    name = "聆心"
    age = 25
    height = 1.75
    is_student = True
    
    print(f"字符串: name = '{name}'")
    print(f"整数: age = {age}")
    print(f"浮点数: height = {height}")
    print(f"布尔值: is_student = {is_student}")
    
    # 类型转换
    print("\n类型转换示例:")
    print(f"str(123) = '{str(123)}'")
    print(f"int('456') = {int('456')}")
    print(f"float('3.14') = {float('3.14')}")

def section_2_lists():
    """1.2 列表操作"""
    print("\n" + "=" * 50)
    print("1.2 列表操作")
    print("=" * 50)
    
    # 创建列表
    fruits = ['苹果', '香蕉', '橙子', '葡萄']
    numbers = [1, 2, 3, 4, 5]
    
    print(f"水果列表: {fruits}")
    print(f"数字列表: {numbers}")
    
    # 访问元素
    print(f"\n第一个元素: fruits[0] = '{fruits[0]}'")
    print(f"最后一个元素: fruits[-1] = '{fruits[-1]}'")
    print(f"切片 fruits[1:3] = {fruits[1:3]}")
    
    # 修改列表
    fruits.append('西瓜')
    print(f"\n添加元素后: {fruits}")
    
    fruits.remove('香蕉')
    print(f"删除元素后: {fruits}")
    
    # 列表长度
    print(f"\n列表长度: len(fruits) = {len(fruits)}")

def section_3_dictionaries():
    """1.3 字典操作"""
    print("\n" + "=" * 50)
    print("1.3 字典操作")
    print("=" * 50)
    
    # 创建字典
    person = {
        'name': '小明',
        'age': 20,
        'city': '北京',
        'skills': ['Python', 'OpenCV', 'Machine Learning']
    }
    
    print(f"字典内容: {person}")
    print(f"访问 name: person['name'] = '{person['name']}'")
    print(f"访问 skills: person['skills'] = {person['skills']}")
    
    # 添加新键值对
    person['gender'] = '男'
    print(f"\n添加 gender 后: {person}")
    
    # 遍历字典
    print("\n遍历字典:")
    for key, value in person.items():
        print(f"  {key}: {value}")

def section_4_functions():
    """1.4 函数定义"""
    print("\n" + "=" * 50)
    print("1.4 函数定义")
    print("=" * 50)
    
    def greet(name, greeting='你好'):
        """打招呼函数"""
        return f"{greeting}, {name}!"
    
    print(f"greet('小红') = '{greet('小红')}'")
    print(f"greet('小李', '嗨') = '{greet('小李', '嗨')}'")
    
    # 带返回值的函数
    def calculate_sum(numbers):
        """计算列表中所有数字的和"""
        total = 0
        for num in numbers:
            total += num
        return total
    
    print(f"\ncalculate_sum([1, 2, 3, 4, 5]) = {calculate_sum([1, 2, 3, 4, 5])}")

def section_5_classes():
    """1.5 类和对象"""
    print("\n" + "=" * 50)
    print("1.5 类和对象")
    print("=" * 50)
    
    class Person:
        """人类"""
        def __init__(self, name, age):
            self.name = name
            self.age = age
        
        def introduce(self):
            """自我介绍"""
            return f"我叫{self.name}，今年{self.age}岁。"
    
    # 创建对象
    person1 = Person('张三', 25)
    person2 = Person('李四', 30)
    
    print(f"person1.introduce() = '{person1.introduce()}'")
    print(f"person2.introduce() = '{person2.introduce()}'")
    
    # 修改属性
    person1.age = 26
    print(f"\n修改年龄后: person1.introduce() = '{person1.introduce()}'")

def section_6_file_io():
    """1.6 文件读写"""
    print("\n" + "=" * 50)
    print("1.6 文件读写")
    print("=" * 50)
    
    import os
    
    # 写入文件
    content = "这是一个测试文件\n第二行内容\n第三行内容"
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("[OK] 文件写入成功: test.txt")
    
    # 读取文件
    with open('test.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"\n文件内容:\n{content}")
    
    # 删除测试文件
    os.remove('test.txt')
    print("\n[OK] 测试文件已删除")

def main():
    print("=" * 60)
    print("第1阶段：Python基础回顾")
    print("=" * 60)
    
    section_1_variables()
    section_2_lists()
    section_3_dictionaries()
    section_4_functions()
    section_5_classes()
    section_6_file_io()
    
    print("\n" + "=" * 60)
    print("Python基础回顾完成！")
    print("下一步：运行 02_opencv_basics.py 学习OpenCV")
    print("=" * 60)

if __name__ == '__main__':
    main()