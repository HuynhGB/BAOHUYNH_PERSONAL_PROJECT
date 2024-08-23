class StringUtils:
    def title_line(text):
        textarry = text.split(" ")
        if len(textarry) > 0:
            textarry[0] = textarry[0].title()
            return " ".join(textarry).strip()
        else:
            return text
