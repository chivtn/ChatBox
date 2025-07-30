import pandas as pd
import torch
import regex as re
import string
from underthesea import word_tokenize,text_normalize
from sentence_transformers import SentenceTransformer,util, models
from app import app
from flask import render_template, jsonify,request
import google.generativeai as genai
import time
from google.api_core import exceptions as google_exceptions


# Cấu hình Gemini API
genai.configure(api_key='AIzaSyA6S1fhukRsyEFpMeAY6nZd_wBVXqHmD14')

# Cấu hình model Gemini
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model_gemini = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)


# Hàm mới để tạo câu trả lời từ Gemini
def generate_answer(query, context):
    prompt = f"""
    Dựa vào các thông tin sau đây:
    {context}

    Hãy trả lời câu hỏi: {query}

    Câu trả lời phải:
    - Ngắn gọn, chính xác
    - Chỉ sử dụng thông tin từ context
    - Nếu không có thông tin, hãy nói 'Xin lỗi, tôi không tìm thấy thông tin liên quan'
    """

    max_retries = 1
    retry_delay = 10  # giây

    for attempt in range(max_retries):
        try:
            response = model_gemini.generate_content(prompt)
            return response.text
        except google_exceptions.ResourceExhausted:
            if attempt < max_retries - 1:
                print(f"Quota exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                return "Xin lỗi, hệ thống hiện đang quá tải. Vui lòng thử lại sau."
        except Exception as e:
            return f"Xin lỗi, có lỗi xảy ra: {str(e)}"


word_embedding_model = models.Transformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base", max_seq_length=256)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
corpus_embeddings = torch.load('retrieval_model/corpus_embeddings.pt')
data = pd.read_csv('data.csv')


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u'\U00010000-\U0010ffff'
                           u"\u200d"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\u3030"
                           u"\ufe0f"
                           "]+", flags=re.UNICODE)

def clean_text(text):
    text = text.lower()  # chuyển toàn bộ text về chữ thường
    text = re.sub(emoji_pattern, " ", text)  # loại bỏ icon
    text = re.sub(r'([a-z]+?)\1+', r'\1', text)  # loại bỏ các ký chữ lặp trong từ(aaaaabbbb--> ab)

    # Đảm bảo khoảng trắng trước và sau giữa các dấu câu --> se , ti
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])\s*(\w)", r"\1 \2 \3", text)

    # Xóa bớt các dấu câu lặp lại --> !!!!->!
    text = re.sub(f"([{string.punctuation}])([{string.punctuation}])+", r"\1", text)

    # Loại bỏ các dấu câu hoặc khoảng trắng ở đâu và cuối câu
    text = text.strip()
    while text.endswith(tuple(string.punctuation + string.whitespace)):
        text = text[:-1]
    while text.startswith(tuple(string.punctuation + string.whitespace)):
        text = text[1:]

    # Loại bỏ tất cả các dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Loại bỏ các khoảng trắng dư thừa trong câu
    text = re.sub(r"\s+", " ", text)

    # Tách từ
    text = text_normalize(text)
    text = word_tokenize(text, format="text")

    return text


# def retrieval(query_text, top_k=5):
#     query = clean_text(query_text)
#     query_embedding = model.encode(query, convert_to_tensor=True)
#
#     cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
#     top_results = torch.topk(cos_scores, k=min(top_k * 3, len(corpus_embeddings)))  # lấy dư để lọc trùng
#
#     seen = set()
#     unique_results = []
#
#     for score, idx in zip(top_results[0], top_results[1]):
#         idx = idx.item()
#         title = data['title'][idx].strip()
#         href = data['href'][idx].strip().rstrip('/')
#         key = (title, href)
#         if key not in seen:
#             seen.add(key)
#             unique_results.append((score, title, href))
#         if len(unique_results) >= top_k:
#             break
#
#     # SỬA LẠI PHẦN NÀY - Tạo context từ các kết quả
#     context_lines = []
#     for _, title, href in unique_results:
#         context_lines.append(f"- {title}: {href}")
#     context = "\n".join(context_lines)
#
#     return unique_results, context

# Đọc thêm cột 'content' từ CSV
data = pd.read_csv('data.csv')


def retrieval(query_text, top_k=5):
    query = clean_text(query_text)
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k * 3, len(corpus_embeddings)))

    seen = set()
    unique_results = []
    best_content = None  # Lưu nội dung của kết quả tốt nhất

    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()
        title = data['title'][idx].strip()
        href = data['href'][idx].strip().rstrip('/')
        content = data['content'][idx]  # Lấy nội dung từ cột 'content'

        key = (title, href)
        if key not in seen:
            seen.add(key)
            unique_results.append((score, title, href))

            # Lưu nội dung của kết quả đầu tiên (tốt nhất)
            if best_content is None:
                best_content = content

        if len(unique_results) >= top_k:
            break

    # Trả về top 5 kết quả và nội dung của kết quả tốt nhất
    return unique_results, best_content


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/search', methods=['POST', 'GET'])
def result():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'khong nhan duoc'})

    query = data.get('query')
    results, context = retrieval(query)  # Nhận cả kết quả và context

    try:
        # Thử tạo câu trả lời từ Gemini
        answer = generate_answer(query, context)
    except Exception as e:
        # Fallback khi có lỗi
        print(f"Gemini error: {str(e)}")
        answer = "Xin lỗi, hiện tôi chưa thể trả lời câu hỏi này."

    print(f'Query: {query}')
    print(f'Answer: {answer}')
    print(results)

    return jsonify({
        'answer': answer,
        'results': [{
            'score': round(score.item(), 4),
            'title': title,
            'href': href
        } for score, title, href in results]
    })


if __name__ == '__main__':
    app.run(debug=True)
