import os
import sys
import json
import datetime
import time
sys.path.append(f'{os.getcwd()}/MUDSENSE')
sys.path.append(f'{os.getcwd()}/LLM_code')
import process
import ocr_cleaning
import llm_predict
import inference


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            return json.JSONEncoder.default(self, obj)


'''读jsonl文件'''
def read_jsonl(path):
    result=[]
    with open(path) as f:
        for line in f:
            result.append(json.loads(line))
    return result


'''写jsonl文件'''
def write_jsonl(path,result):
    with open(path, "w", encoding='utf-8') as jsonl_file:
        for item in result:
            json_line = json.dumps(item, ensure_ascii=False, cls=DateEncoder)
            jsonl_file.write(json_line + "\n")


'''筛选key，只剩姓名也删除'''
def select_result(result):
    # 筛选
    selected_result = []
    noneed = ["编号", "描述", "责任人", "措施", "截止日期", "信息", "说明", "资产状态",
              "资产归属", "上午","下午","业务号","序号","单号","金额"]
    for file in result:
        file_name=list(file.keys())[0]
        split_path=file_name.split("/")
        if not (file_name.endswith(".py") or file_name.endswith(".hiv") or file_name.endswith(".sh")
                or file_name.endswith(".rs") or file_name.endswith(".et") or file_name.endswith(".xlsx")
                or "linux" in split_path or "sam" in split_path or "windwos" in split_path):
            cu_results = file[file_name]
            selected_cu_result = []
            for dict in cu_results:
                temp_dict = {}
                for key in dict:
                    flag=0
                    for no in noneed:
                        if (no in key) or (dict[key]=="") or (dict[key]==None):
                            flag=1
                            break
                    if flag==0:
                        temp_dict[key] = dict[key]
                if temp_dict!={} and temp_dict!={'姓名':"周鹏"}:
                    selected_cu_result.append(temp_dict)
            # 去重
            nosame_result=[]
            for cu in selected_cu_result:
                if cu not in nosame_result:
                    nosame_result.append(cu)
            selected_result.append({file_name:nosame_result})
        else:
            selected_result.append(file)
    return selected_result


'''合并4类抽取方法结果并筛选'''
def conbine(llm_result_path,extract_path,re_result,mudsense_result_path):
    # 大模型结果、表格等结果、规则结果
    llm_result=read_jsonl(llm_result_path)
    direct_result=read_jsonl(extract_path)
    mudsense_result=read_jsonl(mudsense_result_path)

    # 拼接最终结果
    final_result=[]
    # 表格等直接结果、对应文件词典索引
    final_result+=direct_result
    direct_filename = {list(direct_result[count].keys())[0]:count for count in range(len(direct_result))}
    # 大模型结果，不会和其他文件重复
    final_result += llm_result
    # re结果判断文件后拼接
    for re_dict in re_result:
        filename=list(re_dict.keys())[0]
        if filename in direct_filename:
            index=direct_filename[filename]
            final_result[index][filename]+=re_dict[filename]
        else:
            final_result.append(re_dict)

    # 筛选
    selected_result=select_result(final_result)
    # 写
    final_path = "result.jsonl"
    write_jsonl(final_path,selected_result)


'''分簇'''
def devide(path):
    re_result=read_jsonl(path)
    # whn结果转为key/value/pos列表并排序，保存所有文件名
    allfile_keys = []
    allfile_values = []
    allfile_pos = []
    allfiles = []
    for i in range(len(re_result)):
        keys = list(re_result[i].keys())
        # 所有敏感信息类型和值
        afile_keys = []
        afile_values = []
        file_reuslt = re_result[i][keys[0]][0]
        for info_key in file_reuslt:
            if type(file_reuslt[info_key]) == str:
                info = [file_reuslt[info_key]]
            else:
                info = file_reuslt[info_key]
            afile_values += info
            for _ in range(len(info)):
                afile_keys.append(info_key)
        # 所有敏感信息位置
        info_pos = re_result[i][keys[1]]
        afile_pos = []
        for p in info_pos:
            if type(p) == int:
                afile_pos.append(p)
            else:
                afile_pos += p
        if afile_pos != []:
            allfiles.append(keys[0])
            zip_abc = zip(afile_pos, afile_keys, afile_values)
            sorted_zip = sorted(zip_abc, key=lambda x: x[0])
            sorted_pos, sorted_keys, sorted_values = zip(*sorted_zip)
            allfile_pos.append(sorted_pos)
            allfile_keys.append(sorted_keys)
            allfile_values.append(sorted_values)

    # 分簇
    rel_re_result = []
    for i in range(len(allfiles)):
        # 一个文件信息
        afile = allfiles[i]
        afile_pos = allfile_pos[i]
        afile_values = allfile_values[i]
        afile_keys = allfile_keys[i]
        # 按距离分簇，小于cu_min且键值无重复分为一簇，大于则另起一簇
        cu_dist = 50
        file_resul = []
        pre_pos = afile_pos[0]
        a_dict = {afile_keys[0]: afile_values[0]}
        for i in range(1, len(afile_pos)):
            if afile_pos[i] - pre_pos < cu_dist and afile_keys[i] not in a_dict:
                a_dict[afile_keys[i]] = afile_values[i]
            else:
                file_resul.append(a_dict)
                a_dict = {afile_keys[i]: afile_values[i]}
            pre_pos = afile_pos[i]
        file_resul.append(a_dict)
        rel_re_result.append({afile: file_resul})
    return rel_re_result


'''关联认证敏感信息抽取'''
def predict(test_path):
    # try:
        #  LLM for code
    start_t = time.perf_counter()
    llm_result = './llm_result.jsonl'
    prompt = './LLM_code/prompt.txt'
    llm_predict.predict(root_directory=test_path,
                        result_path=llm_result,
                        prompt_path=prompt)
    print("*******************  LLM耗时 %s  *******************" % str(time.perf_counter() - start_t))

    # 处理数据，直接抽取结果存在extract_path，其他识别结果存在text_path
    start_t = time.perf_counter()
    extract_path="extract_result.jsonl"
    text_path="text_result.jsonl"
    process.predict(input_path=test_path,extract_path=extract_path,text_path=text_path)
    print("*******************  ocr+格式数据耗时 %s  *******************" % str(time.perf_counter() - start_t))

    #  输入纯文本，规则预测
    start_t = time.perf_counter()
    re_result_path='extract_ocr_result.jsonl'
    ocr_cleaning.Extract_term(input_path=text_path,
                                output_path=re_result_path)
    print("*******************  规则耗时 %s  *******************" % str(time.perf_counter() - start_t))

    # 输入纯文本，mudsense预测
    start_t = time.perf_counter()
    mudsense_result_path="mudsense_result.jsonl"
    inference.predict(input_path=text_path,output_path=mudsense_result_path)
    print("*******************  MUDSENSE耗时 %s  *******************" % str(time.perf_counter() - start_t))

    start_t = time.perf_counter()
    # 规则结果分簇
    rel_re_result = devide(re_result_path)
    # 4类结果合并、筛选、写,大模型结果+表格等结果+规则结果+mudsense结果
    conbine(llm_result, extract_path, rel_re_result, mudsense_result_path)
    print("DONE!!!!!!!!!!!!!!")
    print("*******************  分簇+合并筛选耗时 %s  *******************" % str(time.perf_counter() - start_t))
    # except:
    #     print("NO!!!!!!!!!!!!!!")
    #     sys.exit(0)

    # 删除临时文件
    os.remove(llm_result)
    os.remove(extract_path)
    os.remove(re_result_path)
    os.remove(mudsense_result_path)
    os.remove(text_path)


if __name__=="__main__":
    start_t=time.perf_counter()
    # 测试集路径
    test_path = "./data"
    predict(test_path=test_path)
    print("*******************  总耗时 %s  *******************"%str(time.perf_counter()-start_t))

















# #  输入纯文本，规则预测
# text_path = "text_result.jsonl"
# re_result_path = 'extract_ocr_result.jsonl'
# ocr_cleaning.Extract_term(input_path=text_path,
#                           output_path=re_result_path)


# # 处理数据，直接抽取结果存在extract_path，其他识别结果存在text_path
# extract_path="extract_result.jsonl"
# text_path="text_result.jsonl"
# test_path = "./data"
# process.predict(input_path=test_path,extract_path=extract_path,text_path=text_path)
