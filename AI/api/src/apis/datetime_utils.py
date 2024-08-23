from datetime import datetime, date, timedelta, time
from word2number import w2n


class DatetimeUtils:

    def extract_time(timeArray=[]):
        hour = 0
        minute = 0
        meridiem = None
        strHour = strMinute = "0"

        nextTokenIndex = len(timeArray)

        if "at" in timeArray:
            atIndex = timeArray.index("at")
            if "on" in timeArray and timeArray.index("on") < nextTokenIndex:
                nextTokenIndex = timeArray.index("on")
            if "in" in timeArray and timeArray.index("in") < nextTokenIndex:
                nextTokenIndex = timeArray.index("in")
            if "next" in timeArray and timeArray.index("next") < nextTokenIndex:
                nextTokenIndex = timeArray.index("next")

            atStr = timeArray[atIndex + 1 : nextTokenIndex]

            if len(atStr) == 1:
                if atStr[0][len(atStr[0]) - 1] == ".":
                    atStr[0] = atStr[0][:-1]

                if "am" in atStr[0] or "a.m" in atStr[0]:
                    meridiem = "am"
                    atStr[0] = atStr[0].replace("am", "").replace("a.m", "")
                if "pm" in atStr[0] or "p.m" in atStr[0]:
                    meridiem = "pm"
                    atStr[0] = atStr[0].replace("pm", "")
                    atStr[0] = atStr[0].replace("p.m", "")

                atStr[0] = atStr[0].strip()
                hm = atStr[0].split(".")
                if len(hm) >= 1:
                    strHour = hm[0]
                if len(hm) >= 2:
                    strMinute = hm[1]
            elif len(atStr) >= 2:
                if atStr[1][len(atStr[1]) - 1] == ".":
                    atStr[1] = atStr[1][:-1]

                if "am" == atStr[1] or "a.m" in atStr[1]:
                    meridiem = "am"
                if "pm" == atStr[1] or "p.m" in atStr[1]:
                    meridiem = "pm"

                atStr[0] = atStr[0].strip()
                hm = atStr[0].split(".")
                if len(hm) >= 1:
                    strHour = hm[0]
                if len(hm) >= 2:
                    strMinute = hm[1]

        hour = int(strHour) if strHour.isdecimal() else 0
        minute = int(strMinute) if strMinute.isdecimal() else 0

        if meridiem == "pm":
            hour = hour + 12

        print("At: {}:{}".format(hour, minute))

        if hour == 0 and minute == 0:
            return None
        else:
            return time(hour=hour, minute=minute)

    def extract_duration(textarry=[]):
        isInFound = False
        inIndex = None
        hourIndex = None
        minuteIndex = None
        for idx, x in enumerate(textarry):
            if isInFound:
                if x != "on" and x != "at" and x != "next":
                    if x == "hours" or x == "hour":
                        hourIndex = idx
                    elif x == "minutes" or x == "minute":
                        minuteIndex = idx
                else:
                    break  # if out of scope of in token, goto another token => break

            if x == "in":
                isInFound = True
                inIndex = idx

        hours = 0
        minutes = 0

        if hourIndex is not None and hourIndex - 1 > inIndex:
            hours = (
                float(textarry[hourIndex - 1])
                if DatetimeUtils.is_number_tryexcept(textarry[hourIndex - 1])
                else 0
            )
            if hours == 0:
                hourWords = " ".join(textarry[inIndex + 1 : hourIndex])
                try:
                    hours = w2n.word_to_num(hourWords)
                except:
                    hours = 0

        if minuteIndex is not None and minuteIndex - 1 > inIndex:
            minutes = (
                int(textarry[minuteIndex - 1])
                if textarry[minuteIndex - 1].isdecimal()
                else 0
            )
            if minutes == 0:
                if hourIndex is not None:
                    minuteWords = " ".join(textarry[hourIndex + 1 : minuteIndex])
                    try:
                        minutes = w2n.word_to_num(minuteWords)
                    except:
                        minutes = 0
                else:
                    minuteWords = " ".join(textarry[inIndex + 1 : minuteIndex])
                    try:
                        minutes = w2n.word_to_num(minuteWords)
                    except:
                        minutes = 0

        if hours != 0 or minutes != 0:
            delta = timedelta(hours=hours, minutes=minutes)
            print("Duration: {}".format(delta))
            return delta
        else:
            return None

    def extract_date(textarry=[]):
        days = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]

        isNextFound = False
        day = None
        nextday = None
        for idx, x in enumerate(textarry):
            if isNextFound:
                if x != "on" and x != "at" and x != "in":
                    if x in days:
                        day = x  # ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                        # Python starts with 0 = Monday
                        today = date.today()
                        weekday = today.weekday()
                        sunday = today + timedelta(days=6 - weekday)
                        nextday = sunday + timedelta(days=1)
                        while days[nextday.weekday()] != day:
                            nextday = nextday + timedelta(days=1)
                else:
                    break  # if out of scope of in token, goto another token => break

            if x == "next":
                isNextFound = True

        if day is not None:
            result = "{}/{}/{}".format(nextday.day, nextday.month, nextday.year)
            print("Date: ".format(result))
            return nextday
        else:
            months = [
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ]
            isOnFound = False
            indexMonth = 0
            for idx, x in enumerate(textarry):
                if isOnFound:
                    if x != "next" and x != "at" and x != "in":
                        if x in months:
                            month = (
                                months.index(x) + 1
                            )  # x will be a month in this months array ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
                            indexMonth = idx

                            dayNum = textarry[indexMonth - 1]
                            # day will be previous item base on indexMonth
                            day = int(dayNum) if dayNum.isdecimal() else None

                            firstDays = ["1st", "2nd", "3rd"]
                            if day is None:
                                if (
                                    textarry[indexMonth - 1] == "of"
                                    and textarry[indexMonth - 2] in firstDays
                                ):
                                    day = firstDays.index(textarry[indexMonth - 2]) + 1
                            thisYear = date.today().year

                            result = "{}/{}/{}".format(day, month, thisYear)
                            print("Date: {}".format(result))
                            return date(day=day, month=month, year=thisYear)

                        elif x in days:
                            today = date.today()
                            weekday = today.weekday()
                            if days.index(x) <= weekday:
                                meetingday = today + timedelta(
                                    days=weekday - days.index(x)
                                )

                                result = "{}/{}/{}".format(
                                    meetingday.day, meetingday.month, meetingday.year
                                )
                                print("Date: ".format(result))

                                return meetingday
                        elif x == "tomorrow":
                            today = date.today()
                            today = date(
                                day=today.day, month=today.month, year=today.year
                            )
                            tomorrow = today + timedelta(days=1)

                            result = "{}/{}/{}".format(
                                tomorrow.day, tomorrow.month, tomorrow.year
                            )
                            print("Date: ".format(result))

                            return tomorrow

                    else:
                        break  # if out of scope of in token, goto another token => break

                if x == "on":
                    isOnFound = True

            print("{}/{}//{}".format(day, 0, 0))

        for idx, x in enumerate(textarry):
            if x in days:
                day = x  # ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                # Python starts with 0 = Monday
                result = date.today()
                weekday = result.weekday()

                dayIdx = days.index(day)
                if dayIdx >= weekday:
                    while days[result.weekday()] != day:
                        result = result + timedelta(days=1)

                    return result
        if len(textarry) == 1 and textarry[0] == "tomorrow":
            result = date.today() + timedelta(days=1)
            return result

        return None

    def enhance_text(text, toType):
        if not isinstance(text, str):
            return ""
        text = text.strip().lower()

        if text[-1] == ".":
            text = text[:-1]

        text = text.replace("?", "")

        if toType == "boolean":
            text = text.replace(".", "")

        mylist = text.split(" ")
        mylist = list(dict.fromkeys(mylist))
        text = " ".join(mylist)
        return text

    def is_number_tryexcept(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
