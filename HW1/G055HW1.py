from pyspark import SparkContext, SparkConf
import sys
import os


def create_pairs(row, S):
    pairs_dict = {}
    rowFields = row.split(',')

    productID = rowFields[1]
    customerID = int(rowFields[6])
    quantity = int(rowFields[3])
    country = rowFields[7]
    
    if quantity > 0 and (country == S or S == "all"):
        pairs_dict[(productID, customerID)] = 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]


def productCustomerDistinctPairs(rowData, S):
    productCustomer = (rowData.flatMap(lambda x: create_pairs(x, S))
                      .groupByKey()
                      .flatMap(lambda x: [(x[0])] ))
    return productCustomer


def gather_pairs_partitions(pairs):
	pairs_dict = {}
	for p in pairs:
		productID = p[0]
		if productID not in pairs_dict.keys():
			pairs_dict[productID] = 1
		else:
			pairs_dict[productID] += 1
	return [(key, pairs_dict[key]) for key in pairs_dict.keys()]


def Popularity1(productCustomer):
    productPopularity1 = (productCustomer.mapPartitions(gather_pairs_partitions)
                         .groupByKey()
                         .mapValues(lambda x: sum(x)))
    return productPopularity1


def Popularity2(productCustomer):
    productPopularity2 = (productCustomer.map(lambda x: (x[0], 1))
                         .reduceByKey(lambda x, y: x + y))
    return productPopularity2


def TopPopularity(productPopularity, H):
    topPopularity = (productPopularity.sortBy(lambda x: x[1], ascending=False)
                    .take(H))
    print(f'Top {H} Products and their Popularities')
    for element in topPopularity:
        print(f'Product {element[0]} Popularity {element[1]};', end=' ')
    print('', end='\n')


def LexicographicOrder(productPopularity):
    ordered = productPopularity.sortByKey().collect()
    for element in ordered:
        print(f'Product: {element[0]} Popularity: {element[1]};', end=' ')
    print('', end='\n')


def main():

    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 5, "Usage: python G055HW1.py <K> <H> <S> <file_name>"

    # SPARK SETUP
    conf = SparkConf().setAppName('G055HW1.py').setMaster("local[*]")
    sc = SparkContext(conf = conf)

    # INPUT READING

    # 1. Read parameters
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    H = sys.argv[2]
    assert H.isdigit(), "H must be an integer"
    H = int(H)

    S = str(sys.argv[3])

    # 2. Read input file and subdivide it into K random partitions
    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path, minPartitions = K).cache()
    rawData.repartition(numPartitions = K)

    print('Number of rows =', rawData.count())

    # COMPUTING (Product, Customer) PAIRS
    productCustomer = productCustomerDistinctPairs(rawData, S)
    print('Product-Customer Pairs =', productCustomer.count())

    # COMPUTING (Product, Popularity) PAIRS

    # 1. Using: mapPartition, groupByKey, mapValues
    productPopularity1 = Popularity1(productCustomer)
    
    # 2. Using: map, reduceByKey
    productPopularity2 = Popularity2(productCustomer)

    if H > 0:
        TopPopularity(productPopularity1, H)
    if H == 0:
        print("productPopularity1: ")
        LexicographicOrder(productPopularity1)
        print("productPopularity2: ")
        LexicographicOrder(productPopularity2)


if __name__ == "__main__":
    main()
