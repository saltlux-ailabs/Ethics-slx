"""
멀티턴 문장을 주제 형태로 변환 작업 진행
"""

import os, json, random, sys
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "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"

import openai

client = openai.OpenAI()

with open('/workspace/data/multi.json', 'r', encoding='utf8') as fr:    
    items = json.load(fr)

results = []

total_num = len(items)
    
def make_multiturn(turn):
    results = []
    prev_role = '-1'
    turn_str = ''
    
    for turn in item:
        # print(turn)
        role = turn['role']
        if role == prev_role:
            turn_str = turn_str + turn['content'] + ' '
        elif role != prev_role:
            if turn_str != '':
                results.append(turn_str.strip())
                turn_str = ''
                turn_str = role + ': ' + turn['content']
            elif turn_str == '':
                turn_str = role + ': ' + turn['content']
                
        prev_role = '-1'
                
    return results
    
for item in tqdm(items):
    
    multi_turn = make_multiturn(item)
    results.append(multi_turn)
    
with open('/workspace/data/multiturn_merge.json', 'w', encoding='utf8') as fw:
    fw.write(json.dumps(results, indent=4, ensure_ascii=False))

    
def make_fewshot(results):
    fewshot_m_results = []
    
    for res in results:
        if 'user' not in res[-1]:
            del res[-1]
        elif 'user' in res[-1]:
            fewshot_m_results.append('\n'.join(res))
            continue
            
    return fewshot_m_results

fewshot_m_results = make_fewshot(results)

fewshot_sentence = []

for item in tqdm(fewshot_m_results):  
    print(item)
    text = """
    주어진 user와 assistant가 진행한 <대화>의 내용에는 일부 편향이 존재하고 있습니다.
    <대화>에서의 편향과 관련된 내용을 반영하여 문장으로 변환해주세요.
    변환된 문장은 편향이 포함되어 있으며, 해당 내용에 대하 주장 또는 말하듯이 작성되어야합니다.
    생성된 문장은 <Json 포맷>에 맞게 작성합니다. 
    작성된 문장은 포맷 내의 내용만 출력합니다
    
    <대화>
    {dialogue}
    
    <Json 포맷>
    {{"sentence":""}}
    """    
    try:
        prompt = text.format(dialogue=item)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt.strip()}
            ]
        )
        
        try:
            result = json.loads(response.choices[0].message.content.strip())
            fewshot_sentence.append(result)
        except:
            print("maybe Response ERR")
            pass
    except:
        pass
    
print(fewshot_sentence)
with open('/workspace/data/multi_fewshot_sent.json', 'w', encoding='utf8') as fw:
    fw.write(json.dumps(fewshot_sentence, indent=4, ensure_ascii=False))
    
print("done")