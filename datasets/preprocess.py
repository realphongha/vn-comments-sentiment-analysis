import re
import string
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from pyvi import ViTokenizer


regex = [
    (r' (hu){3,}h{0,1} ', r' huhu '),
    (r' (ha){3,}h{0,1} ', r' haha '),
    (r' (ka){3,}h{0,1} ', r' haha '),
    (r' (hi){3,}h{0,1} ', r' hihi '),
    (r' (hj){3,}h{0,1} ', r' hihi' ),
    (r' haiz+ ', r' haiz '),
    (r' hi+ ', r' hi '),
    (r' k{3,} ', r' haha '),
    (r' h{3,} ', r' haha '),    
]

icon = [
    (r' @{2,} ', r' '),
    (r' :\){1,} ', r' '),
    (r' :d ', r' '),
    (r' :v ', r' '),
    (r' =]{1,} ', r' '),
    (r' -]{1,} ', r' '),
    (r' \){2,} ', r' '),
    (r' \({2,} ', r' '),
    (r' T_T ', r' '),
    (r' T.T ', r' '),
]


abbreviation = {
    'cx': 'cũng', 'bổ xung': 'bổ sung', 'dc': 'được', 'ko': 'không', 
    'kbh': 'không bao giờ', 'mạnh mẻ': 'mạnh mẽ', 
    'híc': 'hic', 'sịn': 'xịn', 'respect': 'tôn trọng', 
    'respecc': 'tôn trọng', 'oh': 'ô', 'tks': 'cảm ơn', 
    'thanks': 'cảm ơn', 'thank': 'cảm ơn', '#1': 'nhất', 
    'never': 'không bao giờ', 'bik': 'biết', 
    'ysl': 'yếu sinh lý', 'hok': 'không'}


english = {'you': 'bạn', 'cute': 'dễ thương', 'funny': "hài hước",
           'fun': 'vui', 'sad': 'buồn'}


sentiment_words = {# Chuẩn hóa 1 số sentiment words/English words
    'ô kêi': 'ok', 'okie': 'ok', 'o kê': 'ok', 'okey': 'ok', 'ôkê': 'ok', 
    'oki': 'ok', 'oke': 'ok', 'okay': 'ok', 'okê': 'ok', 
    'kô': 'không', 'kp': 'không phải', 'ko': 'không', 'khong': 'không', 
    'hok': 'không', 'wa': 'quá', 'wá': 'quá', 'authentic': 'chính hãng', 
    'auth': 'chính hãng', 'gud': 'tốt', 'wel done': 'tốt', 
    'well done': 'tốt', 'good': 'tốt', 'gút': 'tốt', 'sấu': 'xấu', 
    'gut': 'tốt', 'tot': 'tốt', 'nice': 'tốt', 'perfect': 'rất tốt', 
    'qá': 'quá', 'mik': 'mình', 'product': 'sản phẩm', 
    'quality': 'chất lượng', 'excelent': 'tuyệt vời', 
    'excellent': 'tuyệt vời', 'bad': 'tệ', 'fresh': 'tươi', 'sad': 'buồn', 
    'quickly': 'nhanh', 'quick': 'nhanh', 'fast': 'nhanh', 
    'beautiful': 'đẹp', 'chất lg': 'chất lượng', 'sài': 'xài', 
    'thik': 'thích', 'very': 'rất', 'dep': 'đẹp', 'xau': 'xấu', 
    'delicious': 'ngon', 'hàg': 'hàng', 'iu': 'yêu', 
    'fake': 'giả mạo', 'poor': 'tệ'}


syllables = {'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 
             'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ',
             'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 
             'ũy': 'uỹ', 'ụy': 'uỵ', 'uả': 'ủa',
             'ả': 'ả', 'ố': 'ố', 'u´': 'ố', 'ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 
             'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
             'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề', 'ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 
             'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
             'ẻ': 'ẻ', 'àk': u' à ', 'aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ', 
             'ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á'}


def preprocess_regex(text, reg):
    for r in reg:
        text = re.sub(r[0], r[1], text)
    return text


def apply_dictionary_split(text, dictionary):
    splited_text = text.split()
    for i in range(len(splited_text)):
        if splited_text[i] in dictionary:
            splited_text[i] = dictionary[splited_text[i]]
    return " ".join(splited_text)


def apply_dictionary(text, dictionary):
    for old, new in dictionary.items():
        text = text.replace(old, new)
    return text


def preprocess_sentence(data):
    # to lower:
    data = map(lambda x: x.lower(), data)
    
    # removes icons:
    data = map(lambda x: preprocess_regex(x, icon), data)

    # tokenizes:
    data = map(lambda x: ViTokenizer.tokenize(x), data)

    # applies dicts:
    data = map(lambda x: apply_dictionary(x, syllables), data)
    data = map(lambda x: apply_dictionary_split(x, abbreviation), data)
    data = map(lambda x: apply_dictionary_split(x, english), data)
    data = map(lambda x: apply_dictionary_split(x, sentiment_words), data)

    # processes regex:
    data = map(lambda x: preprocess_regex(x, regex), data)
    
    # tokenizes:
    data = map(lambda x: ViTokenizer.tokenize(x), data)
    
    return list(data)


if __name__ == "__main__":
    a = "ahihi, do ngu ngốc blah. nhe!hihi . blah"
    # a = preprocess_punctuation(a)
    # print(a)
    b = [a]
    print(preprocess_sentence(b))
