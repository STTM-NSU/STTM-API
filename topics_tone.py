from collections import defaultdict


def select_words_of_topic_word_distributions(topic_word_distributions, threshold):
    selected_topic_word_distributions = defaultdict(dict)
    for hour, topic_word_distribution in topic_word_distributions.items():
        for topic, words in topic_word_distribution.items():
            C = 0
            count = 0
            for word, value in words:
                C += value
                count += 1
                if C >= threshold:
                    selected_topic_word_distributions[hour][topic] = words[:count]
                    # print(f'{hour}:{topic}:   {count}   :{C}')
                    break
            if C < threshold:
                selected_topic_word_distributions[hour][topic] = words[:count]
                # print(f'{hour}:{topic}:   {count}   :{C}')
    return selected_topic_word_distributions


def get_topics_tone(topic_word_distributions, word_tone, threshold):
    topics_tone = defaultdict(dict)
    selected_words = select_words_of_topic_word_distributions(topic_word_distributions, threshold)
    for hour, selected_words_hour in selected_words.items():

        for topic, words in selected_words_hour.items():
            positive_score = 0.0
            negative_score = 0.0
            normalization = 0.0

            for word, weight in words:
                tone = word_tone.get(word, 0)
                if tone == 0:
                    continue

                contribution = tone * weight
                normalization += abs(contribution)

                if tone > 0:
                    positive_score += contribution
                else:
                    negative_score -= contribution

            p_prob = positive_score / normalization if positive_score else 0.0
            n_prob = negative_score / normalization if negative_score else 0.0
            tone_score = p_prob - n_prob

            topics_tone[hour][topic] = tone_score

            # print(f'hour: {hour}, topic: {topic}, pProb: {p_prob:.4f}, nProb: {n_prob:.4f}, result: {tone_score:.4f}')

    return topics_tone

