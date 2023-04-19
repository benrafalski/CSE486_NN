from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer




def sentiment_vader(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"
  
    return negative, neutral, positive, compound, overall_sentiment


class Sentence():
    def __init__(self, sent, rating):
        self.sentence = sent
        self.rating = rating

sentences = [
    "2I am unable to connect to the router using an Ethernet cable as I do not have one in my wall. I do not know what WPS is and the manual does not describe this part. The manual does not describe where to search for these instruction on my device.",
    "5This manual was super easy to use and understand. I was easily able to connect to the Wifi using my ethernet connection.",
    "1This manual does a horrible job of accounting for any troubleshooting that people may need. My router was plugged in but the LED was not on so it was not recieving power. No where in the manual does it give troubleshooting tips. I had to spend hours to even get the power on. This manual is worthless.",
    "4Using the manual I was able to easily set up my WiFi. I wish the sections were split up a bit different so it was easier to read.",
    "3I was able to set up my WiFI using the manual. It took me a while to find the WiFi password since it is never mentioned where the router label is located.",
    "5I recently purchased the NETGEAR CENTRIA manual and I am extremely impressed with it. The manual is easy to follow and provides step-by-step instructions on how to set up and use the CENTRIA. The illustrations are clear and the text is concise, making it easy to understand even for someone who is not tech-savvy. Overall, I highly recommend this manual for anyone who needs guidance on how to use the NETGEAR CENTRIA.",
    "4The NETGEAR CENTRIA manual is a great resource for anyone who is looking to set up their CENTRIA. The manual is well-written and provides a lot of detail about the device's features and functionality. The only downside is that it can be a bit overwhelming for those who are not familiar with networking and computer hardware. However, with a bit of patience and perseverance, anyone can use this manual to get their CENTRIA up and running.",
    "3The NETGEAR CENTRIA manual is an okay resource for those who are looking to set up their device. The manual provides basic instructions on how to connect the CENTRIA to a network and access its features. However, the manual is a bit outdated and doesn't cover all of the device's features in detail. Additionally, the illustrations are not very clear, which can make it difficult to follow some of the instructions.",
    "2I was disappointed with the NETGEAR CENTRIA manual. The manual is poorly written and difficult to follow. The illustrations are not clear and the instructions are not very detailed. Additionally, the manual does not cover all of the device's features, which can make it frustrating for users who are looking for specific information.",
    "1The NETGEAR CENTRIA manual is terrible. The manual is poorly written, the illustrations are not clear, and the instructions are confusing. I would not recommend this manual to anyone. If you are looking to set up your CENTRIA, I suggest finding another resource that is more user-friendly and easier to follow.",
    "5The manual on Setting Up a Wireless Access List by MAC Address is an excellent resource. It provides clear, step-by-step instructions on how to set up the access list and ensure that only authorized devices can connect to the wireless network. The illustrations and screenshots are helpful and easy to follow, and the language is straightforward and jargon-free. Overall, I highly recommend this manual for anyone who needs to set up a wireless access list.",
    "4The manual on Setting Up a Wireless Access List by MAC Address is a great resource for those who need to set up an access list. The instructions are clear and easy to follow, and the illustrations are helpful in guiding you through the process. However, the manual could benefit from more examples and troubleshooting tips, especially for users who may encounter issues while setting up the access list.",
    "3The manual on Setting Up a Wireless Access List by MAC Address is an okay resource for those who need to set up an access list. The instructions are clear, but the manual could use more details and examples to make the process easier to follow. Additionally, the manual may not be very helpful for users who are not familiar with networking or wireless technology.",
    "2I found the manual on Setting Up a Wireless Access List by MAC Address to be confusing and not very helpful. The instructions are not very clear, and the illustrations are not helpful in guiding you through the process. The manual assumes that the user has a lot of prior knowledge of networking and wireless technology, which can make it difficult for beginners to understand.",
    "1I was very disappointed with the manual on Setting Up a Wireless Access List by MAC Address. The instructions were confusing and difficult to follow, and the illustrations were not helpful at all. I would not recommend this manual to anyone, especially those who are not very familiar with networking or wireless technology.",
    "5The manual on setting up a fixed IPv6 internet connection is an excellent resource. The instructions are clear and easy to follow, and the illustrations are helpful in guiding you through the process. The manual also provides troubleshooting tips for common issues that may arise during setup. Overall, I highly recommend this manual for anyone who needs to set up a fixed IPv6 internet connection.",
    "4The manual on setting up a fixed IPv6 internet connection is a great resource for those who need to set up a connection. The instructions are clear and concise, and the illustrations are helpful in guiding you through the process. However, the manual could benefit from more examples and troubleshooting tips, especially for users who may encounter issues while setting up the connection.",
    "3The manual on setting up a fixed IPv6 internet connection is an okay resource for those who need to set up a connection. The instructions are clear, but the manual could use more details and examples to make the process easier to follow. Additionally, the manual may not be very helpful for users who are not familiar with networking or internet technology.",
    "2I found the manual on setting up a fixed IPv6 internet connection to be confusing and not very helpful. The instructions are not very clear, and the illustrations are not helpful in guiding you through the process. The manual assumes that the user has a lot of prior knowledge of networking and internet technology, which can make it difficult for beginners to understand.",
    "1I was very disappointed with the manual on setting up a fixed IPv6 internet connection. The instructions were confusing and difficult to follow, and the illustrations were not helpful at all. I would not recommend this manual to anyone, especially those who are not very familiar with networking or internet technology.",
    "5The HP printer manual is an excellent resource for anyone who needs help setting up or using their printer. The manual is well-organized, easy to understand, and includes helpful illustrations and screenshots. I was able to follow the instructions and set up my printer in no time. Overall, I highly recommend this manual.",
    "4The HP printer manual is a great resource for those who need help setting up or using their printer. The instructions are clear and concise, and the illustrations are helpful in guiding you through the process. However, the manual could benefit from more troubleshooting tips for common issues that may arise during use.",
    "3The HP printer manual is an okay resource for those who need help setting up or using their printer. The instructions are clear, but the manual could use more details and examples to make the process easier to follow. Additionally, the manual may not be very helpful for users who are not very familiar with printers or technology.",
    "2I found the HP printer manual to be confusing and not very helpful. The instructions are not very clear, and the illustrations are not helpful in guiding you through the process. The manual assumes that the user has a lot of prior knowledge of printers and technology, which can make it difficult for beginners to understand.",
    "1I was very disappointed with the HP printer manual. The instructions were confusing and difficult to follow, and the illustrations were not helpful at all. I would not recommend this manual to anyone, especially those who are not very familiar with printers or technology."
]

test_data = [Sentence(s[1:], s[:1]) for s in sentences]

for t in test_data:
    print(f'Rating: {t.rating}/5\nPrediction: {sentiment_vader(t.sentence)}\nSentence: {t.sentence}\n')