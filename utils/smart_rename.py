



def smart_rename(expression, text, counter=0):
    """
    Returns the expression text specially parsed. It can contain two types of expression,
    counter term and slice term. The counter term can only be in python number format,
    defined here https://pyformat.info/. Eg. {:04d} - counter will be formatted and this term
    will be replaced by the number formatted in this way, eg. if counter = 25, {:04d} will
    be replaced by 0025.

    Other type is slice expression {start_index:end_index}. This term will be replaced by
    the text[start_index:end_index] value. It is the python slicing format for lists/strings/arrays.

    If {} brackets need to be in the resulting string, backslash must be used before the bracket,
    indicating it is not the term.

    ---------Example------

    expression = r'\{{:03d}\}: t = {1:} us'

    text = #164.6

    counter = 39

    result = r'{039}: t = 164.6 us'

    :param expression:
        str
    :param text:
        str
    :param counter:
        int
    :return:
        str
    """

    # copied from https://stackoverflow.com/questions/680826/python-create-slice-object-from-string
    def parse_slice(v):
        """
        Parses text like python "slice" expression (ie ``-10::2``).

        :param v:
            the slice expression or a lone integer
        :return:
            - None if input is None/empty
            - a ``slice()`` instance (even if input a lone numbrt)
        :raise ValueError:
            input non-empty but invalid syntax
        """
        orig_v = v
        v = v and v.strip()
        if not v:
            return

        try:
            if ':' not in v:
                ## A lone number given.
                v = int(v)
                return slice(v, v + 1)

            return slice(*map(lambda x: int(x.strip()) if x.strip() else None,
                              v.split(':')))
        except:
            raise ValueError("Parsing error of slice {}.".format(orig_v))

        # ## An alternative is to return `slice(None)` here.
        # raise trt.TraitError("Syntax-error in '%s' slice!" % orig_v)

    try:
        start_idx = 0
        contents = []

        for i, char in enumerate(expression):
            if char == '{' and i != 0 and expression[i - 1] != '\\':
                start_idx = i

            if char == '}' and expression[i - 1] != '\\':
                contents.append([start_idx, i + 1, expression[start_idx + 1:i]])

        if len(contents) == 0:
            return expression.replace('\\{', '{').replace('\\}', '}')

        # print the begining
        result = expression[:contents[0][0]]

        for i, con in enumerate(contents):

            # for different formats for numbers, look https://pyformat.info/
            if 'd' in con[2] or 'f' in con[2]:
                content = ("{" + con[2] + "}").format(counter)
            else: # slicing text
                content = text[parse_slice(con[2])]

            result += content + (expression[con[1]:contents[i + 1][0]] if i < len(contents) - 1 else '')

        # print the last part
        result += expression[contents[-1][1]:]

        return result.replace('\\{', '{').replace('\\}', '}')
    except:
        raise ValueError("Parsing error of expression {}.".format(expression))


#
# names = ['4865', '44555', "#4686486", "4adads"]
#
# # expression = "\{{:03d}\}: t = {1:} us"
# expression = "4865"
#
#
# for i in range(len(names)):
#     print(names[i], "     formatted to      ", smart_rename(expression, names[i], i))
#
