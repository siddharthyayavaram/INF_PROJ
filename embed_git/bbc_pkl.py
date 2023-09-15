import csv
import pickle
import time

def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)

def save_arr(l):
    with open('bbc_data.pkl', 'wb') as f:
        pickle.dump(l, f)

def main():

    csv_file_path = 'bbc-news-data.csv'
    csv_data = []

    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            csv_data.append([str(cell) for cell in row])

    first_row = csv_data[1]
    print("First Row:", first_row)

    print(len(csv_data))

    final=[]

    for i in range(len(csv_data)):
        concat = "".join(csv_data[i])
        lst = concat.split("\t")
        for j in range(len(lst)):
            if j!=0 and j%3==0:
                final.extend(lst[j].split(". "))


    print(len(final))

    l = [string for string in final if ( not has_numbers(string) )]

    # save_arr(l)

if __name__=="__main__":
    start = time.time()
    main()
    end = time.time()
    print("The time of execution of above program is :",
        (end-start) * 10**3, "ms")