"""
Function takes the output of the MRDA dataset and
converts it into a .csv format, after cleaning up the
text input.
"""

import sys
import os
import pandas as pd
import numpy as np
import re

classes = {'D': 'Disruption',
           'B': 'Backchannel',
           'F': 'Filler',
           'S': 'Statement',
           'Q': 'Question',
           'Z': 'Non-labelled'}

def fixT(text):
    return text.replace("t ", "t")

def fixM(text):
    return text.replace("m ", "m")

def fixApostrophe(text):
    return text.replace(" \'", "\'")

def readRawIntoString(filename):
    with open(filename, 'r') as file:
        output = " ".join(file.read().replace('\n', '').split())
    return output

def splitPipebar(label):
    split = label.split('|')
    return split[0], split[1]

def disPresent(label):
    return any([tag in label for tag in ['%', '%-', '%--', "x"]])

def statePresent(label):
    return any([tag in label for tag in ['s', 'br']])

def questPresent(label):
    return any([tag in label for tag in ['q']])

def fillPresent(label):
    return any([tag in label for tag in ['f', 'h']])

def backPresent(label):
    return any([tag in label for tag in ['b']])

def nonPresent(label):
    return any([tag in label for tag in ['z']])

def applyMRDAClassMap(label):

    #1. Pipebar present
    if '|' in label:
        #2. Disruption present
        if disPresent(label):
            left, right = splitPipebar(label)
            #3. If filler in left side of pipebar
            if fillPresent(left):
                if statePresent(right):
                    return classes['S']
                if questPresent(right):
                    return classes['Q']
                if disPresent(right):
                    return classes['D']
            #3. No filler in left side of pipebar
            else:
                return classes['D']

        #2. No disruption present
        else:
            if statePresent(label):
                return classes['S']
            if questPresent(label):
                return classes['Q']
            if fillPresent(label):
                return classes['F']
            if backPresent(label):
                return classes['B']
            if nonPresent(label):
                return classes['Z']

    #1. No pipebar present
    else:
        # 2. Disruption present
        if disPresent(label):
            return classes['D']
        # 2. No disruption present
        else:
            if statePresent(label):
                return classes['S']
            if questPresent(label):
                return classes['Q']
            if fillPresent(label):
                return classes['F']
            if backPresent(label):
                return classes['B']
            if nonPresent(label):
                return classes['Z']

    return "FAILED"

def utterDicts(text, dialogueCounter):
    out = []
    # Extract tags from raw text input
    groupsTag = re.findall("([a-zA-Z]{2}\d{3}) (\[.*?\])", text)
    groupsText = re.findall("(\: .*?)[a-zA-Z]{2}\d{3}", text)
    # Since we miss the last one, here is a dirty fix
    groupsText.append(re.compile("\[.*?\]").split(text)[-1])

    actorCol = [i[0] for i in groupsTag]
    daLabel = [i[1] for i in groupsTag]

    # Replace actor-tags by alphabetical letters (for clarity)
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    actorMap = {key:value for (key, value) in zip(list(set(actorCol)),
                    [alph[i] for i in range(len(set(actorCol)))])}
    actorCol = list(map(lambda x: actorMap[x], actorCol))

    # Apply MRDA-class map for general tags (5 classes)
    daLabelConv = list(map(lambda x: applyMRDAClassMap(x), daLabel))

    for idx in range(len(groupsTag)):
        dict = {"Actor": actorCol[idx],
                "Dialogue Act": daLabelConv[idx],
                "Dialogue ID": dialogueCounter,
                "Utterance": groupsText[idx]}
        out.append(dict)

    return out

if __name__ == '__main__':
    raw_folder = "MRDA_as_txt/"
    list_of_files = os.listdir(raw_folder)

    table_of_dicts = []
    dialogueCounter = 0

    # Loop through all files
    for file in list_of_files:
        raw_string = readRawIntoString(raw_folder + file)

        # Perform fix needed to split tags and text
        out = fixT(raw_string)
        out = fixM(out)
        out = fixApostrophe(out)

        # return a list of one dict per utterance with keys
        # 'Actor', 'Dialogue Act', 'Dialogue ID' and 'Utterance'
        convDict = utterDicts(out, dialogueCounter)
        dialogueCounter += 1
        table_of_dicts += convDict

    dataframe = pd.DataFrame(table_of_dicts)
    dataframe.to_csv("../../CSVData/MRDA.csv", index = False)

