import json
import re
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

fr = open('valid.json', 'r', encoding='utf-8')

VOCAB = './vocab.txt'
tokenizer = BertTokenizer.from_pretrained(VOCAB)


chinese_entity_type_vs_english_entity_type = {
    '图书作品': 'People',
    '学科专业': 'Industry',
    '景点': 'Business',
    '历史人物': 'Report',
    '生物': 'Organization',
    '网络小说': 'Risk',
    '电视综艺': 'Article',
    '歌曲': 'Indicator',
    '机构': 'Brand',
    '行政区': 'Product',
    '企业': '18',
    '影视作品': '17',
    '国家': '15',
    '书籍': '14',
    '人物': '13',
    '地点': '11',
    "音乐专辑": '1',
    "城市": '2',
    "Text": '3',
    "气候": '4',
    "Date": '5',
    "语言": '6',
    "Number": '7',
    "出版社": '8',
    "网站": '9',
    "目": '10',
    "学校": '12',
    "作品": '16'
}
english_entity_type_vs_chinese_entity_type = {v: k for k, v in chinese_entity_type_vs_english_entity_type.items()}

START_TAG = "[CLS]"
END_TAG = "[SEP]"
O = "O"
B1 = "B-1"
I1 = "I-1"
B2 = "B-2"
I2 = "I-2"
B3 = "B-3"
I3 = "I-3"
B4 = "B-4"
I4 = "I-4"
B5 = "B-5"
I5 = "I-5"
B6 = "B-6"
I6 = "I-6"
B7 = "B-7"
I7 = "I-7"
B8 = "B-8"
I8 = "I-8"
B9 = "B-9"
I9 = "I-9"
B10 = "B-10"
I10 = "I-10"
B11 = "B-11"
I11 = "I-11"
B12 = "B-12"
I12 = "I-12"
B13 = "B-13"
I13 = "I-13"
B14 = "B-14"
I14 = "I-14"
B15 = "B-15"
I15 = "I-15"
B16 = "B-16"
I16 = "I-16"
B17 = "B-17"
I17 = "I-17"
B18 = "B-18"
I18 = "I-18"
BPeople = "B-People"
IPeople = "I-People"
BIndustry = "B-Industry"
IIndustry = "I-Industry"
BBusiness = 'B-Business'
IBusiness = 'I-Business'
BProduct = 'B-Product'
IProduct = 'I-Product'
BReport = 'B-Report'
IReport = 'I-Report'
BOrganization = 'B-Organization'
IOrganization = 'I-Organization'
BRisk = 'B-Risk'
IRisk = 'I-Risk'
BArticle = 'B-Article'
IArticle = 'I-Article'
BIndicator = 'B-Indicator'
IIndicator = 'I-Indicator'
BBrand = 'B-Brand'
IBrand = 'I-Brand'

PAD = "[PAD]"
UNK = "[UNK]"
tag2idx = {
    START_TAG: 0,
    END_TAG: 1,
    O: 2,
    BPeople: 3,
    IPeople: 4,
    BIndustry: 5,
    IIndustry: 6,
    BBusiness: 7,
    IBusiness: 8,
    BProduct: 9,
    IProduct: 10,
    BReport: 11,
    IReport: 12,
    BOrganization: 13,
    IOrganization: 14,
    BRisk: 15,
    IRisk: 16,
    BArticle: 17,
    IArticle: 18,
    BIndicator: 19,
    IIndicator: 20,
    BBrand: 21,
    IBrand: 22,
    B1: 23,
    I1: 24,
    B2 :  25,
    I2 :  26,
    B3 :  27,
    I3 :  28,
    B4 :  29,
    I4 :  30,
    B5 :  31,
    I5 :  32,
    B6 :  33,
    I6 :  34,
    B7 :  35,
    I7 :  36,
    B8 :  37,
    I8 :  38,
    B9 :  39,
    I9 :  40,
    B10 :  41,
    I10 :  42,
    B11 :  43,
    I11 :  44,
    B12 :  45,
    I12 :  46,
    B13 :  47,
    I13 :  48,
    B14 :  49,
    I14 :  50,
    B15 :  51,
    I15 :  52,
    B16 :  53,
    I16 :  54,
    B17 :  55,
    I17 :  56,
    B18 :  57,
    I18 :  58,
    PAD: 59,
    UNK: 60,
}
tag2id = tag2idx
idx2tag = {v: k for k, v in tag2idx.items()}

# {
# "sent": "半导体行情的风险是什么",
# "sent_tokens": ["半", "导", "体", "行", "情", "的", "风", "险", "是", "什", "么"],
# "sent_token_ids": [1288, 2193, 860, 6121, 2658, 4638, 7599, 7372, 3221, 784, 720],
# "entity_labels": [{"entity_type": "研报", "start_token_id": 0, "end_token_id": 10, "start_index": 0, "end_index": 10,
# "entity_tokens": ["半", "导", "体", "行", "情", "的", "风", "险", "是", "什", "么"],
# "entity_name": "半导体行情的风险是什么"}],
# "tags": ["B-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report"],
# "tag_ids": [11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]},

# {"tokens": ["歌", "曲", "《", "春", "天", "的", "悲", "剧", "》", "选", "自", "周", "传", "雄", "专", "辑", "《", "发", "觉", "》", "，", "由", "离", "云", "作", "词", "，", "周", "传", "雄", "（", "小", "刚", "）", "作", "曲", "，", "1994", "年", "歌", "林", "唱", "片", "出", "版", "并", "发", "行"],
#  "spo_list": [["春天的悲剧", "所属专辑", "发觉"], ["春天的悲剧", "歌手", "周传雄"], ["春天的悲剧", "作词", "离云"]],
#  "spo_details": [[3, 8, "歌曲", "所属专辑", 17, 19, "音乐专辑"], [3, 8, "歌曲", "歌手", 27, 30, "人物"], [3, 8, "歌曲", "作词", 22, 24, "人物"]]}

for line in fr:
    sentdic_list = []
    instence = json.loads(line)

    for index, ins in enumerate(instence):
        if index == 400:
            break
        sentdic = {}

        sentdic['tokens'] = ins['tokens']
        # sentdic['sent_token_ids'] = tokenizer.convert_tokens_to_ids(ins['tokens'])

        # sent_tokens = sentdic['sent_tokens']
        # for index, i in enumerate(ins['sent_tokens']):
        #     if i == ' ' or i == '\uf020':
        #         ins['sent_tokens'][index] = '-'
        #
        # sentdic['sent_token_ids_'] = tokenizer.convert_tokens_to_ids(ins['sent_tokens'])

        entity_labels = []
        entities = []

        spo_list = ins['spo_list']
        spo_details = ins['spo_details']

        for index, ent in enumerate(spo_details):
            ent1 = spo_list[index][0]
            ent2 = spo_list[index][2]

            entity1_label = {}
            entity2_label = {}

            if ent1 not in entities:
                entities.append(ent1)

                entity1_label['start'] = ent[0]
                entity1_label['end'] = ent[1]
                entity1_label['type'] = ent[2]
                # entity1_label['entity_name'] = ent1
                entity_labels.append(entity1_label)

            if ent2 not in entities:
                entities.append(ent2)

                entity2_label['start'] = ent[4]
                entity2_label['end'] = ent[5]
                entity2_label['type'] = ent[6]
                # entity2_label['entity_name'] = ent2
                entity_labels.append(entity2_label)

        # BIOES标签
        # if not entity_labels:
        #     tags = [O for _ in range(len(sent_tokens))]
        #     tag_ids = [tag2idx[O] for _ in range(len(sent_tokens))]
        # else:
        #     tags = []
        #     tag_ids = []
        #     for sent_token_index in range(len(sent_tokens)):
        #         tag = O
        #         for entity_label in entity_labels:
        #             if sent_token_index == entity_label['start_index']:
        #                 tag = f'B-{chinese_entity_type_vs_english_entity_type[entity_label["entity_type"]]}'
        #             elif entity_label['start_index'] < sent_token_index < entity_label["end_index"]:
        #                 tag = f'I-{chinese_entity_type_vs_english_entity_type[entity_label["entity_type"]]}'
        #         tag_id = tag2idx[tag]
        #         tags.append(tag)
        #         tag_ids.append(tag_id)
        # assert len(sent_tokens) == len(tags) == len(tag_ids)
        #
        # sentdic['tags'] = tags
        # sentdic['tag_ids'] = tag_ids
        # sentdic['entites'] = entities
        sentdic['entites'] = entity_labels

        sentdic_list.append(sentdic)

typedic_list = {}
type_dicts = {}
for i in chinese_entity_type_vs_english_entity_type.keys():
    type_dict = {}
    type_dict['short'] = i
    type_dict['verbose'] = i
    type_dicts[i] = type_dict
typedic_list['entities'] = type_dicts


with open('type.json', 'w', encoding='utf-8') as file_obj:
    # json.dump(sentdic_list, file_obj, ensure_ascii=False)
    json.dump(typedic_list, file_obj, ensure_ascii=False)

print('保存成功')

