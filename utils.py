import os
import csv
import json

def get_id(root="OpenBG-IMG", c="train"):
    entities = set()
    relations = set()
    
    f_path = os.path.join(root, root+'_'+c+'.tsv')
    with open(f_path, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            
            entities.add(line[0])
            entities.add(line[2])
            relations.add(line[1])
            
    num_entities = len(entities)
    num_relations = len(relations)
    
    print(num_entities)
    print(num_relations)
    entities_id = [i for i in range(num_entities)]
    relations_id = [i for i in range(num_relations)]
    
    entities2id = dict(zip(entities, entities_id))
    relations2id = dict(zip(relations, relations_id))
    
    with open(os.path.join(root,'entities2id.json'), 'w') as f:
        json.dump(entities2id, f)
    with open(os.path.join(root, 'relations2id.json'), 'w') as f:
        json.dump(relations2id, f)
    
    return entities2id, relations2id

def sets2id(entities2id, relations2id, root="OpenBG-IMG", sets='train'):
    id_sets = []
    
    with open(os.path.join(root, root+'_'+sets+'.tsv'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            ids = [entities2id[line[0]], relations2id[line[1]], entities2id[line[2]]]
            id_sets.append(ids)
    
    with open(os.path.join(root, sets+'_id.tsv'), 'w', newline="") as f:
        tsv_w = csv.writer(f, delimiter="\t")
        tsv_w.writerows(id_sets)




if __name__ == "__main__":
    en2id, rel2id = get_id()
    sets2id(entities2id=en2id, relations2id=rel2id)