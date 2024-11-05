# parallel processing

import re
from concurrent.futures import ThreadPoolExecutor

# Updated regex pattern to match 'syllabus' even if embedded within other words
syllabus_pattern = re.compile(r'\bsyllabus\b', re.IGNORECASE)
per_curiam_pattern = re.compile(r'Per Curiam', re.IGNORECASE)

def remove_before_syllabus(text):
    match = syllabus_pattern.search(text)
    if match:
        return text[match.end():]
    return text

def remove_spaces(text):
    return re.sub(r'\s+', '', text)

def remove_syllabus_from_case(case_text, syllabus_text):
    # Remove everything before and including "syllabus"
    syllabus_text = remove_before_syllabus(syllabus_text)
    case_text = remove_before_syllabus(case_text)

    # Remove all spaces from both texts
    syllabus_text_no_spaces = remove_spaces(syllabus_text)
    case_text_no_spaces = remove_spaces(case_text)

    # Find the longest matching substring of syllabus in case text
    max_match_length = 0
    max_match_end = 0
    for i in range(len(syllabus_text_no_spaces)):
        for j in range(i + 1, len(syllabus_text_no_spaces) + 1):
            substring = syllabus_text_no_spaces[i:j]
            if substring in case_text_no_spaces:
                if len(substring) > max_match_length:
                    max_match_length = len(substring)
                    max_match_end = case_text_no_spaces.find(substring) + len(substring)

    # Remove the longest matching substring from case text
    case_text_no_spaces = case_text_no_spaces[max_match_end:]

    # Restore spaces in case text
    restored_text = []
    mod_index = 0
    for char in case_text:
        if mod_index < len(case_text_no_spaces) and char.lower() == case_text_no_spaces[mod_index].lower():
            restored_text.append(char)
            mod_index += 1
        elif char.isspace():
            restored_text.append(char)

    # Remove leading and trailing spaces
    return ''.join(restored_text).strip()

def remove_double_spaces(text):
    return ' '.join(text.split())

def process_case(item, subkey, compiled_phrases_end):
    if 'justia_sections' in item and subkey in item['justia_sections']:
        case_text = item['justia_sections'][subkey].replace('\n', ' ')

        if 'Syllabus' in item['justia_sections']:
            syllabus_text = item['justia_sections']['Syllabus'].replace('\n', ' ')
            case_text = remove_syllabus_from_case(case_text, syllabus_text)

        match_end = next((phrase.search(case_text) for phrase in compiled_phrases_end if phrase.search(case_text)), None)
        if match_end:
            item['opinion_end_phrase'] = match_end.group()
            end = case_text.find(item['opinion_end_phrase'])
            if end != -1:
                opinion_text = case_text[:end + len(item['opinion_end_phrase'])].strip()
                item['opinionOfTheCourt'] = remove_double_spaces(opinion_text)
            return item, 'with_phrases'
        else:
            if per_curiam_pattern.search(case_text):
                return item, 'with_per_curiam'
            else:
                return item, 'without_phrases'
    return item, 'without_phrases_end'

def FindCasesWithPhrase(data, subkey, phrases_end):
    cases_with_phrases = []
    cases_without_phrases = []
    cases_with_per_curiam = []
    cases_without_phrases_end = []

    compiled_phrases_end = [re.compile(re.escape(phrase), re.IGNORECASE) for phrase in phrases_end]

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda item: process_case(item, subkey, compiled_phrases_end), data)

    for item, category in results:
        if category == 'with_phrases':
            cases_with_phrases.append(item)
        elif category == 'with_per_curiam':
            cases_with_per_curiam.append(item)
        elif category == 'without_phrases':
            cases_without_phrases.append(item)
        else:
            cases_without_phrases_end.append(item)

    return cases_with_phrases, cases_without_phrases, cases_with_per_curiam, cases_without_phrases_end

cases_with_phrases_parallel, cases_without_phrases, cases_with_per_curiam, cases_without_phrases_end = FindCasesWithPhrase(data, subkey, phrases_end)

print(cases_with_phrases_parallel[0]['opinionOfTheCourt'])


count_cases_with_phrases = len(cases_with_phrases_parallel)
count_cases_without_phrases = len(cases_without_phrases)
count_cases_with_per_curiam = len(cases_with_per_curiam)
count_cases_without_phrases_end = len(cases_without_phrases_end)

print(f"Number of cases with 'Case' subkey under 'justia_sections' containing end phrases: {count_cases_with_phrases}")
print(f"Number of cases with 'Case' subkey under 'justia_sections' not ˙µ˜' end phrases: {count_cases_without_phrases}")
print(f"Number of cases with 'Case' subkey under 'justia_sections' containing 'Per Curiam': {count_cases_with_per_curiam}")
print(f"Number of cases with 'Case' subkey under 'justia_sections' not containing phrases_end: {count_cases_without_phrases_end}")


cases_parallel = FilterCases(cases_with_phrases_parallel)
df_parallel = convert_cases_to_dataframe(cases_parallel)
convert_and_upload_data(cases_parallel,'FILTERED_parallel')



# ############################



# new 
from concurrent.futures import ThreadPoolExecutor

syllabus_pattern = re.compile(r'syllabus', re.IGNORECASE)
per_curiam_pattern = re.compile(r'Per Curiam', re.IGNORECASE)

def remove_before_syllabus(text):
    match = syllabus_pattern.search(text)
    if match:
        return text[match.end():]
    return text

def remove_spaces(text):
    return re.sub(r'\s+', '', text)

def find_longest_match(syllabus_text_no_spaces, case_text_no_spaces):
    max_match_length = 0
    max_match_end = 0
    for i in range(len(syllabus_text_no_spaces)):
        for j in range(i + 1, len(syllabus_text_no_spaces) + 1):
            substring = syllabus_text_no_spaces[i:j]
            if substring in case_text_no_spaces:
                if len(substring) > max_match_length:
                    max_match_length = len(substring)
                    max_match_end = case_text_no_spaces.find(substring) + len(substring)
    return max_match_end

def remove_syllabus_from_case(case_text, syllabus_text):
    syllabus_text = remove_before_syllabus(syllabus_text)
    case_text = remove_before_syllabus(case_text)

    syllabus_text_no_spaces = remove_spaces(syllabus_text)
    case_text_no_spaces = remove_spaces(case_text)

    max_match_end = find_longest_match(syllabus_text_no_spaces, case_text_no_spaces)
    case_text_no_spaces = case_text_no_spaces[max_match_end:]

    restored_text = []
    mod_index = 0
    for char in case_text:
        if mod_index < len(case_text_no_spaces) and char.lower() == case_text_no_spaces[mod_index].lower():
            restored_text.append(char)
            mod_index += 1
        elif char.isspace():
            restored_text.append(char)

    return ''.join(restored_text).strip()

def remove_double_spaces(text):
    return ' '.join(text.split())

def process_case(item, subkey, compiled_phrases_end):
    if 'justia_sections' in item and subkey in item['justia_sections']:
        case_text = item['justia_sections'][subkey].replace('\n', ' ')

        if 'Syllabus' in item['justia_sections']:
            syllabus_text = item['justia_sections']['Syllabus'].replace('\n', ' ')
            case_text = remove_syllabus_from_case(case_text, syllabus_text)

        match_end = next((phrase.search(case_text) for phrase in compiled_phrases_end if phrase.search(case_text)), None)
        if match_end:
            item['opinion_end_phrase'] = match_end.group()
            end = case_text.find(item['opinion_end_phrase'])
            if end != -1:
                opinion_text = case_text[:end + len(item['opinion_end_phrase'])].strip()
                item['opinionOfTheCourt'] = remove_double_spaces(opinion_text)
            return item, 'with_phrases'
        else:
            if per_curiam_pattern.search(case_text):
                return item, 'with_per_curiam'
            else:
                return item, 'without_phrases'
    return item, 'without_phrases_end'

def FindCasesWithPhrase(data, subkey, phrases_end):
    cases_with_phrases = []
    cases_without_phrases = []
    cases_with_per_curiam = []
    cases_without_phrases_end = []

    compiled_phrases_end = [re.compile(re.escape(phrase), re.IGNORECASE) for phrase in phrases_end]

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda item: process_case(item, subkey, compiled_phrases_end), data)

    for item, category in results:
        if category == 'with_phrases':
            cases_with_phrases.append(item)
        elif category == 'with_per_curiam':
            cases_with_per_curiam.append(item)
        elif category == 'without_phrases':
            cases_without_phrases.append(item)
        else:
            cases_without_phrases_end.append(item)

    return cases_with_phrases, cases_without_phrases, cases_with_per_curiam, cases_without_phrases_end

# Example usage
subkey = 'Case'


# Assuming `data` is your dataset
cases_with_phrases_FAST_3, cases_without_phrases, cases_with_per_curiam, cases_without_phrases_end = FindCasesWithPhrase(data, subkey, phrases_end)
print(cases_with_phrases_FAST_3[0]['opinionOfTheCourt'])


count_cases_with_phrases = len(cases_with_phrases_FAST_3)
count_cases_without_phrases = len(cases_without_phrases)
count_cases_with_per_curiam = len(cases_with_per_curiam)
count_cases_without_phrases_end = len(cases_without_phrases_end)

print(f"Number of cases with 'Case' subkey under 'justia_sections' containing end phrases: {count_cases_with_phrases}")
print(f"Number of cases with 'Case' subkey under 'justia_sections' not containing end phrases: {count_cases_without_phrases}")
print(f"Number of cases with 'Case' subkey under 'justia_sections' containing 'Per Curiam': {count_cases_with_per_curiam}")
print(f"Number of cases with 'Case' subkey under 'justia_sections' not containing phrases_end: {count_cases_without_phrases_end}")


cases_FAST_3 = FilterCases(cases_with_phrases_FAST_3)
df_FAST_3 = convert_cases_to_dataframe(cases_FAST_3)
convert_and_upload_data(cases_FAST_3,'FILTERED_FAST_3')
