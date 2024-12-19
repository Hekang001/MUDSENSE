import json
import re

# 读取 JSON 文件

pattern0= r'\n[\u4e00-\u9fff0-9]*：\n\n[a-zA-Z0-9*]+?[\n)，]'  #处理冒号后有\n的情况
pattern1=r'[（，\n：][\u4e00-\u9fffa-zA-Z0-9]+：[a-zA-Z0-9@*]+?[\n)，）（]'
# # str1='''\n密码： 学校邮箱密码（初始密码： 000000 ）\n如有问题请联系：\n电话： 22238800                Email ： serv@xxx.edu.cn\n'''
# # matches = re.findall(pattern6, str1)
# # print(matches)

pattern2=r'[\n|，|：][\u4e00-\u9fff]*[是|为][a-zA-Z0-9@.*”“]+[\n|，|)|）]' #处理。。。是。。类型
pattern3=r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}' #处理邮箱
pattern4=r'(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' #处理IP
pattern5=r'https?://\S+' #处理网址
pattern6=r'周鹏|张三|刘振华' #处理姓名
pattern7=r'\n[\u4e00-\u9fff]*[电话][\u4e00-\u9fff]*\n\n[0-9*]+?[\n）)，]'


result=[]
def Extract_term(input_path,output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析 JSON 对象
            json_data = json.loads(line)
            file_name= list(json_data.keys())[0]
            data=json_data[file_name]
            data = data.replace("     ", "\n\n")
            data=data.replace(" ","")
            data=data.replace("\n","\n\n")
            # 在每一行中搜索匹配的文本
            matches = re.finditer(pattern1, data)
            matches0 = re.finditer(pattern0,data)
            matches2=re.finditer(pattern2,data)
            matches3=re.finditer(pattern3,data)
            matches4 = re.finditer(pattern4, data)
            matches5 = re.finditer(pattern5, data)
            name_matches=re.finditer(pattern6, data)
            matches7=re.finditer(pattern7, data)

            wash=[]
            for item in matches:
                start, end = item.span()
                item=item.group()
                start_index=start
                item=item.replace('\n','')
                item=item.replace('，','')
                item = item.replace('(', '')
                item = item.replace(')', '')
                item = item.replace('（', '')
                item = item.replace('）', '')
                item = "：".join(item.split("：")[1:]) if len(item.split("：")) > 2 else item
                item=item+"#"+str(start_index)

                wash.append(item)

            for item in matches0:
                start, end = item.span()
                item = item.group()
                start_index = start
                item=item.replace('\n','')
                item=item.replace('，','')
                item = item.replace('(', '')
                item = item.replace(')', '')
                item = item.replace('（', '')
                item = item.replace('）', '')
                item = item + "#" + str(start_index)
                wash.append(item)
            for item in matches2:
                start, end = item.span()
                item = item.group()
                start_index = start
                item=item.replace('\n','')
                item=item.replace('，','')
                item = item.replace('(', '')
                item = item.replace(')', '')
                item = item.replace('（', '')
                item = item.replace('）', '')
                item=item.replace('是', '：')
                item=item.replace('为', '：')
                item = item + "#" + str(start_index)
                wash.append(item)
            for item in matches4:
                start, end = item.span()
                item = item.group()
                start_index = start
                item = "IP地址：" + item
                item = item + "#" + str(start_index)
                wash.append(item)
            for item in matches3:
                start, end = item.span()
                item = item.group()
                start_index = start
                item="邮箱："+item
                item = item + "#" + str(start_index)
                wash.append(item)
            for item in matches5:
                start, end = item.span()
                item = item.group()
                start_index = start
                item="网址："+item
                item = item + "#" + str(start_index)
                wash.append(item)

            #姓名 把他改成find
            for item in name_matches:
                start, end = item.span()
                item = item.group()
                start_index = start
                item="姓名："+item
                item = item + "#" + str(start_index)
                wash.append(item)

            for item in matches7:
                start, end = item.span()
                item = item.group()
                start_index = start
                item = item.replace('\n\n', '：')
                item = item.replace('\n', '')
                item = item.replace('，', '')
                item = item + "#" + str(start_index)
                wash.append(item)

            result_dict={}

            for item in wash:
                key, value = item.split("：", 1)  # 使用冒号分割键值对，只分割一次
                key = key.strip()
                key = key.replace('是', '')
                key = key.replace('为', '')
                value = value.strip()


                if key not in result_dict:
                    result_dict[key] = value

                else:
                    # 如果键已经存在，将值添加到列表中
                    if isinstance(result_dict[key], list):
                        result_dict[key].append(value)
                    else:
                        if result_dict[key] != value:
                            result_dict[key] = [result_dict[key], value]
            #加起始位置

            start_index=[]
            for key, value in result_dict.items():
                if isinstance(value, list):
                    list_index=[]
                    for i in range(len(value)):
                        # 处理列表中的每个元素
                        key_value, pos = value[i].split("#", 1)
                        list_index.append(int(pos))
                        value[i]=key_value
                    start_index.append(list_index)
                else:
                    key_value, pos = value.split("#", 1)
                    start_index.append(int(pos))
                    result_dict[key] = key_value

            res={
                file_name:[result_dict],
                #新加的没用东西,抽取信息的起始位置
                'pos':start_index
            }
            json_string = json.dumps(res, ensure_ascii=False)
            result.append(json_string)
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in result:
            file.write(item+'\n')


def process_ocr(input_path,output_path):
    Extract_term(input_path, output_path)


if __name__ == "__main__":
    input_path=r"D:\pythonproject\pythonProject\starcoder\file_list\pptx_docx_text.jsonl"


    output_path=r"D:\pythonproject\pythonProject\starcoder\file_list\output_result.jsonl"

    Extract_term(input_path, output_path)