def classify_sample(preprocessed_text, classifierlist, subclassifier_list, vectorizer, tfidfconverter):
    
    vect = vectorizer.transform([preprocessed_text]).toarray()
    tfidfmatrix = tfidfconverter.transform(vect).toarray()
    classified_probs = []
    
    for i in range(len(classifierlist)):
        probabilities = classifierlist[i].predict_proba(tfidfmatrix)
        classified_probs.append(probabilities)
        
    labelprobs = [j[0][1] for j in classified_probs]
    print(classified_probs)
    binaryprobs = []
    
    if labelprobs[0]<=labelprobs[1] and labelprobs[0]<=labelprobs[2]:
        binaryprobs = subclassifier_list[0].predict_proba(tfidfmatrix)
        binarylabel = subclassifier_list[0].predict(tfidfmatrix)
        return binarylabel[0], binaryprobs[0], labelprobs, 0
    if labelprobs[1]<=labelprobs[0] and labelprobs[1]<=labelprobs[2]:
        binaryprobs = subclassifier_list[1].predict_proba(tfidfmatrix)
        binarylabel = subclassifier_list[1].predict(tfidfmatrix)
        return binarylabel[0], binaryprobs[0], labelprobs, 1
    if labelprobs[2]<=labelprobs[1] and labelprobs[2]<=labelprobs[0]:
        binaryprobs = subclassifier_list[2].predict_proba(tfidfmatrix)
        binarylabel = subclassifier_list[2].predict(tfidfmatrix)
        return binarylabel[0], binaryprobs[0], labelprobs, 2
    
    print("edge case")
    return labelprobs


if __name__ == '__main__':
    url = "https://www.foxnews.com/media/biden-appears-to-check-notes-after-press-ask-about-russia"
    label, scores, all_probs, least_expected = classify_sample("this is the text of an article")
    print(label)
    print(scores)
    print(all_probs)
    print(least_expected)
    
    
# url = "https://www.breitbart.com/politics/2021/07/03/confused-joe-biden-takes-out-notes-to-answer-question-on-russia/"
# url = 'https://www.breitbart.com/clips/2021/07/03/fncs-carlson-kamala-harris-a-power-hungry-buffoon-posing-as-a-competent-adult/'
# url = "https://www.breitbart.com/politics/2021/07/02/kamala-harris-facing-white-house-sabotage-as-establishment-media-lays-out-pete-buttigiegs-path-to-presidency/"
# url = "https://www.breitbart.com/politics/2021/07/02/kamala-harris-facing-white-house-sabotage-as-establishment-media-lays-out-pete-buttigiegs-path-to-presidency/"
# url = "https://www.breitbart.com/politics/2021/07/03/exclusive-nh-school-employee-resigns-citing-anti-white-critical-race-theory-staff-training/"
# url = "https://www.foxnews.com/opinion/biden-economic-crisis-spending-hurt-recovery-rep-ron-estes"



