import numpy as np
import pandas as pd

FILENAME = 'Abominable_VALIDATION_Data_FOR_STUDENTS_v750_2215.csv'
csv = pd.read_csv(FILENAME)

csv['Age'] = csv['Age'].apply( lambda datapoint: 2 * round(datapoint/2) )
csv['Ht'] = csv['Ht'].apply( lambda datapoint: 4 * round(datapoint/4) )
csv['TailLn'] = csv['TailLn'].apply( lambda datapoint: 2 * round(datapoint/2) )
csv['HairLn'] = csv['HairLn'].apply( lambda datapoint: 2 * round(datapoint/2) )
csv['BangLn'] = csv['BangLn'].apply( lambda datapoint: 2 * round(datapoint/2) )
csv['Reach'] = csv['Reach'].apply( lambda datapoint: 2 * round(datapoint/2) )
csv['EarLobes'] = csv['EarLobes'].apply( lambda datapoint: 1 * round(datapoint/1) )

predictions = []

for datapoint in csv.to_numpy():

    # Datapoint values:
    age = datapoint[0]
    ht = datapoint[1]
    tailln = datapoint[2]
    hairln = datapoint[3]
    bangln = datapoint[4]
    reach = datapoint[5]
    earlobes = datapoint[6]

    prediction = -1
    if ( earlobes < 1 ):
        if ( bangln <= 6 ):
            if ( hairln <= 10 ):
                if ( ht <= 136 ):
                    if ( tailln <= 6 ):
                        if ( ht <= 132 ):
                            prediction = 1
                        else:
                            if ( reach <= 140 ):
                                if ( age <= 46 ):
                                    prediction = -1
                                else:
                                    if ( age <= 56 ):
                                        prediction = 1
                                    else:
                                        prediction = -1
                            else:
                                prediction = 1
                    else:
                        if ( reach <= 140 ):
                            if ( age <= 40 ):
                                if ( bangln <= 4 ):
                                    if ( reach <= 138 ):
                                        prediction = -1
                                    else:
                                        if ( age <= 36 ):
                                            prediction = -1
                                        else:
                                            prediction = -1
                                else:
                                    if ( tailln <= 10 ):
                                        if ( ht <= 132 ):
                                            prediction = 1
                                        else:
                                            prediction = -1
                                    else:
                                        if ( reach <= 134 ):
                                            prediction = -1
                                        else:
                                            prediction = -1
                            else:
                                prediction = -1
                        else:
                            if ( bangln <= 4 ):
                                if ( tailln <= 12 ):
                                    prediction = 1
                                else:
                                    prediction = 1
                            else:
                                prediction = 1
                else:
                    prediction = -1
            else:
                if ( age <= 46 ):
                    if ( ht <= 140 ):
                        if ( reach <= 124 ):
                            prediction = -1
                        else:
                            prediction = 1
                    else:
                        if ( tailln <= 14 ):
                            if ( tailln <= 6 ):
                                prediction = 1
                            else:
                                prediction = -1
                        else:
                            if ( age <= 42 ):
                                prediction = 1
                            else:
                                prediction = -1
                else:
                    if ( bangln <= 4 ):
                        if ( tailln <= 16 ):
                            prediction = -1
                        else:
                            prediction = -1
                    else:
                        if ( age <= 68 ):
                            if ( reach <= 136 ):
                                prediction = -1
                            else:
                                if ( ht <= 132 ):
                                    prediction = 1
                                else:
                                    if ( tailln <= 14 ):
                                        if ( ht <= 148 ):
                                            prediction = -1
                                        else:
                                            prediction = -1
                                    else:
                                        prediction = 1
                        else:
                            prediction = -1
        else:
            if ( hairln <= 6 ):
                if ( tailln <= 6 ):
                    prediction = 1
                else:
                    if ( age <= 52 ):
                        if ( ht <= 136 ):
                            prediction = -1
                        else:
                            prediction = -1
                    else:
                        prediction = 1
            else:
                if ( hairln <= 10 ):
                    if ( age <= 52 ):
                        if ( ht <= 132 ):
                            prediction = 1
                        else:
                            if ( age <= 44 ):
                                if ( ht <= 148 ):
                                    if ( age <= 30 ):
                                        prediction = 1
                                    else:
                                        if ( reach <= 150 ):
                                            prediction = 1
                                        else:
                                            prediction = 1
                                else:
                                    if ( hairln <= 8 ):
                                        prediction = -1
                                    else:
                                        prediction = 1
                            else:
                                if ( tailln <= 10 ):
                                    prediction = 1
                                else:
                                    if ( ht <= 152 ):
                                        prediction = 1
                                    else:
                                        prediction = -1
                    else:
                        if ( age <= 64 ):
                            if ( ht <= 136 ):
                                prediction = 1
                            else:
                                prediction = -1
                        else:
                            prediction = 1
                else:
                    prediction = 1
    else:
        if ( hairln <= 8 ):
            if ( bangln <= 4 ):
                if ( tailln <= 14 ):
                    if ( reach <= 138 ):
                        if ( age <= 46 ):
                            if ( tailln <= 8 ):
                                if ( ht <= 132 ):
                                    prediction = 1
                                else:
                                    prediction = -1
                            else:
                                if ( age <= 30 ):
                                    prediction = 1
                                else:
                                    if ( ht <= 132 ):
                                        prediction = -1
                                    else:
                                        prediction = -1
                        else:
                            if ( tailln <= 6 ):
                                prediction = -1
                            else:
                                prediction = -1
                    else:
                        if ( bangln <= 2 ):
                            prediction = -1
                        else:
                            if ( age <= 22 ):
                                prediction = -1
                            else:
                                if ( reach <= 142 ):
                                    if ( age <= 52 ):
                                        if ( tailln <= 4 ):
                                            prediction = 1
                                        else:
                                            prediction = 1
                                    else:
                                        prediction = -1
                                else:
                                    if ( hairln <= 6 ):
                                        if ( ht <= 156 ):
                                            prediction = 1
                                        else:
                                            prediction = -1
                                    else:
                                        if ( age <= 50 ):
                                            prediction = 1
                                        else:
                                            prediction = 1
                else:
                    if ( age <= 56 ):
                        if ( age <= 30 ):
                            prediction = -1
                        else:
                            if ( reach <= 146 ):
                                if ( hairln <= 6 ):
                                    prediction = -1
                                else:
                                    prediction = -1
                            else:
                                if ( ht <= 144 ):
                                    prediction = 1
                                else:
                                    if ( tailln <= 20 ):
                                        if ( age <= 50 ):
                                            prediction = -1
                                        else:
                                            prediction = 1
                                    else:
                                        prediction = 1
                    else:
                        prediction = -1
            else:
                if ( tailln <= 6 ):
                    prediction = 1
                else:
                    if ( reach <= 140 ):
                        if ( age <= 44 ):
                            if ( age <= 34 ):
                                prediction = 1
                            else:
                                if ( reach <= 138 ):
                                    if ( tailln <= 10 ):
                                        prediction = 1
                                    else:
                                        prediction = -1
                                else:
                                    prediction = 1
                        else:
                            if ( age <= 52 ):
                                if ( tailln <= 10 ):
                                    prediction = 1
                                else:
                                    if ( ht <= 128 ):
                                        prediction = -1
                                    else:
                                        prediction = -1
                            else:
                                prediction = -1
                    else:
                        if ( tailln <= 14 ):
                            prediction = 1
                        else:
                            if ( age <= 52 ):
                                if ( ht <= 144 ):
                                    prediction = 1
                                else:
                                    if ( reach <= 150 ):
                                        prediction = -1
                                    else:
                                        if ( ht <= 148 ):
                                            prediction = 1
                                        else:
                                            prediction = 1
                            else:
                                if ( reach <= 156 ):
                                    if ( ht <= 140 ):
                                        prediction = 1
                                    else:
                                        if ( tailln <= 20 ):
                                            prediction = -1
                                        else:
                                            prediction = 1
                                else:
                                    if ( hairln <= 6 ):
                                        prediction = -1
                                    else:
                                        if ( age <= 54 ):
                                            prediction = 1
                                        else:
                                            prediction = 1
        else:
            prediction = 1

    predictions.append(prediction)
    print(prediction)

df = pd.DataFrame(predictions,columns=['ClassID'])
df.to_csv('HW05_Mehboob_Mohammed_MyClassifications.csv',index=False)
