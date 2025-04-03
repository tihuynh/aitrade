from typing import TextIO

with open("penhouse.jpg","rb+") as f:
    # size_to_read = 10
    # f_content = f.readline(size_to_read)x
    # print(f_content)
    # f.seek(1)
    # f_content = f.readline(size_to_read)
    # print(f_content)
    with open("penhouse_copy.jpg","wb+") as wf:
        # for line in f :
        #     wf.write(line)
        chunk_binary = 4096
        # rf_chunk = f.read(chunk_binary)
        # while len(rf_chunk) > 0:
        #     wf.write(rf_chunk)
        #     rf_chunk = f.read(chunk_binary)
        while True:
            rf_chunk = f.read(chunk_binary)
            if not rf_chunk:
                break
            wf.write(rf_chunk)
print("fie sao chep thanh cong")