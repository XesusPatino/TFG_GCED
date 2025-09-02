import argparse
import numpy as np

# Actualizar para ML-1M
RATING_FILE_NAME = dict({'movie': 'ratings.dat', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie': '::', 'book': ';', 'music': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0})


def read_item_index_to_entity_id_file():
    file = 'C:/Users/xpati/Documents/TFG/ml-1m/data-kgnn-ls/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    file = 'C:/Users/xpati/Documents/TFG/ml-1m/data-kgnn-ls/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_ratings = dict()  # Almacenar todas las valoraciones con sus valores originales

    # Para ML-1M no necesitamos saltar la primera línea ya que no tiene encabezado
    start_line = 0 if DATASET == 'movie' else 1
    
    for line in open(file, encoding='utf-8').readlines()[start_line:]:
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        # Para ML-1M, los índices son diferentes
        if DATASET == 'movie':
            user_index_old = int(array[0])
            item_index_old = array[1]
            rating = float(array[2])
        else:
            user_index_old = int(array[0])
            item_index_old = array[1]
            rating = float(array[2])

        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        # Almacenar todas las valoraciones con sus valores originales
        if user_index_old not in user_ratings:
            user_ratings[user_index_old] = []
        user_ratings[user_index_old].append((item_index, rating))

    print('converting rating file ...')
    writer = open('C:/Users/xpati/Documents/TFG/ml-1m/data-kgnn-ls/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    
    for user_index_old, item_rating_list in user_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        # Escribir todas las valoraciones con sus valores originales (1-5)
        for item_index, rating in item_rating_list:
            writer.write('%d\t%d\t%.1f\n' % (user_index, item_index, rating))
    
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('C:/Users/xpati/Documents/TFG/ml-1m/data-kgnn-ls/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
    for line in open('C:/Users/xpati/Documents/TFG/ml-1m/data-kgnn-ls/' + DATASET + '/kg.txt', encoding='utf-8'):
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')