
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return tc_categories(category_i), category_i

def tc_categories(index=-1):
    categories = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]
    if index == -1:
        return categories
    return categories[index]

