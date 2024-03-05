import re


def remove_by_pattern(content: str, pattern: str) -> str:
    content = re.sub(pattern, "", content, flags=re.DOTALL)
    return content


def remove_description(content: str) -> str:
    pattern = r"<description>[\s\S]*?</description>"
    return remove_by_pattern(content, pattern)


def remove_notes(content: str) -> str:
    pattern = r"<body name=\"notes\">[\s\S]*?</body>"
    return remove_by_pattern(content, pattern)


def remove_binaries(content: str) -> str:
    pattern = r"<binary[\s\S]*?</binary>"
    return remove_by_pattern(content, pattern)


def remove_page_numbers(content: str) -> str:
    pattern = r"<p>[\d]*?</p>"
    return remove_by_pattern(content, pattern)


def remove_tags(content: str) -> str:
    pattern = r"<[\s\S]*?>"
    return remove_by_pattern(content, pattern)


def remove_extra_space(content: str) -> str:
    content = "\n".join([line.strip() for line in content.strip().split("\n")])
    content = re.sub(r"\n\n\n\n*", "\n\n", content)
    return content


def replace_irregular_symbols(content: str) -> str:
    for ir_s, s in [
        ("ó", "о"),
        ("é", "е"),
        ("á", "а"),
        ("ё", "е"),
        ("Ё", "Е"),
        ("\xa0", ""),
        ("//", ","),
        ("…", "..."),
        ("»", '"'),
        ("«", '"'),
        ("“", '"'),
        ("„", '"'),
        ("`", ""),
        ("'", '"'),
        ("—", "-"),
        ("&lt;", ""),
        ("&gt;", ""),
        ("&amp;", ""),
        ("-|", ""),
        ("§", ""),
        ("√", "")
    ]:
        content = content.replace(ir_s, s)
    
    content = remove_by_pattern(content, r"\[\d+\]")
    return content


def main():
    with open(
        "data/Strugackiy_Strugackie-Tomirovannyy-i-hronologicheskiy-sbornik-proizvedeniy.S5ERfw.590031.fb2",
        "r",
    ) as file:
        content = file.read()

    content = remove_description(content)
    content = remove_notes(content)
    content = remove_binaries(content)
    content = remove_page_numbers(content)
    content = remove_tags(content)
    content = remove_extra_space(content)
    content = replace_irregular_symbols(content)

    # print(content.split())
    # print('\xa0')
    l = list(set(content))
    l.sort()
    print(l)
    with open("data/Strugackie_prepared.txt", "w") as file:
        file.write(content)


if __name__ == "__main__":
    main()
