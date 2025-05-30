def print_divider(title="구분선", color_code=31):
    print(f"\033[{color_code}m" + "="*100 + f" {title} " + "="*100 + "\033[0m")