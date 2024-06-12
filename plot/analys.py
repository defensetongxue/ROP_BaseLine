import os,json
with  open('./record.json','r') as f:
    orignal_record=json.load(f)
paser_record={}
for k in  orignal_record:
    model_name=k.split('_')[0]
    res_list=orignal_record[k]["result"]
    res={
        "accuracy":0.0,
        "auc":0.0,
        "recall":0.0
    }
    for split_name in res_list:
        res['auc']+=res_list[split_name]['auc']
        res["accuracy"]+=res_list[split_name]["accuracy"]
        res["recall"]+=res_list[split_name]["recall_pos"]
    res['auc']=round(res['auc']/4,4)
    res["accuracy"]=round(res["accuracy"]/4,4)
    res['recall']=round(res["recall"]/4,4)
    paser_record[model_name]=res
with open('./paser_record.json','w') as f:
    json.dump(paser_record,f)