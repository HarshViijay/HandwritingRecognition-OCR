from autocorrect import spell

def correct_sentence(line):
    lines = line.strip().split(' ')
    new_line = ""
    similar_word = {}
    for l in lines:
        new_line += spell(l) + " "
    return new_line



