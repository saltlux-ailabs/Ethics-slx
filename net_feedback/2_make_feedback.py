"""
사용자가 풀이한 내용을 바탕으로 피드백 제공
"""

import requests
import json
from tqdm import tqdm
import random

random.seed(42)

def make_params(prompt_str):
    params = {
    "conv_id": "e79548cfcfd46862",
    "user_prompt": prompt_str,
    "stream": False,
    "knowledge": "string",
    "multiturn": False,
    "history": [
    ],
    "history_count": -1,
    "best_of": 1,
    "frequency_penalty": 0.1,
    "repetition_penalty": 1.1,
    "max_new_tokens": 1024,
    "seed": 0,
    "stop": [],
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "truncate": None,
    "hidden_prompt": "{knowledge}/n/n{user_prompt}",
    "system_prompt": ""
    }

    return params

def inference(user_sheets, uri):
    random.shuffle(user_sheets)
    feed_back_json = []
    
    for user_sheet in user_sheets:
        prompt_str = """
        주어진 문제는 사회/윤리적으로 편향이 있는지 없는지를 판단하기 위한 문제이며, 해당 문제에 대한 답변이 존재합니다.

        주어진 문제와 답변의 풀이가 올바른지 분석해주세요. 만약 풀이가 올바르지 않다면, 해당 문제에 대한 정답 풀이, 해석 그리고 해당 편향에 대해 대응하기 위한 응답을 한국어로 작성해주세요.

        문제:
        {problem}
        
        답변:
        {answer}
        """
        
        prompt_str = prompt_str.format(problem=user_sheet['problem'], answer=user_sheet['user_pred'])
        
        param = make_params(prompt_str)
        response = requests.post(uri, json=param)
        results = json.loads(response.text)
        completion = results['message']
        feed_back_json.append(
            {
                "prompt" : "주어진 문제와 답변의 정답 풀이 및 해설은 다음과 같습니다.\n\n문제: " + user_sheet['problem'] + '\n\n답변: ' + user_sheet['user_pred'],
                "completion" : completion
            }
        )
        
        
    with open('/workspace/data/eval_set/2_feedback_set.json', 'w', encoding='utf8') as fw:
        fw.write(json.dumps(feed_back_json, indent=4, ensure_ascii=False))


if __name__ == '__main__':
  uri = 'http://211.109.9.151:14006/completion'
  conv_id = 'e79548cfcfd46862'
  user_sheet_path = "/workspace/data/user_answer/user_sheet.json"
  
  with open(user_sheet_path, 'r', encoding='utf8') as fr:
      user_sheet = json.load(fr)
  
  inference(user_sheet, uri)