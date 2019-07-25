def alter(file,old_str,new_str):
    file_data = ""
    with open(file, "r") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str,new_str)
            file_data += line
    with open(file,"w") as f:
        f.write(file_data)

alter("/home/jiangyl/PDB/text.txt", "jiangyl/pics", "jiangyl/PDB/pics")
