import ast
import os.path


def generate_pseudo_code(source_code):
    """
    Generates pseudo-code from Python source code.

    Args:
        source_code: The Python source code as a string.

    Returns:
        The pseudo-code representation as a string, or None if parsing fails.
    """

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"SyntaxError: {e}")  # Handle Syntax Errors gracefully
        return None

    pseudo_code = ""

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            pseudo_code += f"{node.name}("
            for i, arg in enumerate(node.args.args):
                pseudo_code += arg.arg
                if i < len(node.args.args) - 1:
                    pseudo_code += ", "
            pseudo_code += ")\n"

            for stmt in node.body:
                pseudo_code += translate_statement(stmt, 1)  # Start indentation at 1
            pseudo_code += "END " + node.name + "()\n"

        elif isinstance(node, ast.ClassDef):  # Handle Class Definitions
            pseudo_code += f"CLASS {node.name}:\n"
            for stmt in node.body:
                pseudo_code += translate_statement(stmt, 1)
            pseudo_code += "END CLASS " + node.name + "\n"

    return pseudo_code


def translate_statement(stmt, indent_level):
    """Translates a single Python statement to pseudo-code."""
    indent = "  " * indent_level  # Consistent indentation

    if isinstance(stmt, ast.Assign):
        target = ""
        for t in stmt.targets:  # Handle multiple assignments
            if isinstance(t, ast.Tuple):
                for i, elt in enumerate(t.elts):
                    target += translate_target(elt)
                    if i < len(t.elts) - 1:
                        target += ", "
            else:
                target = translate_target(t)

        value = translate_expression(stmt.value)
        return f"{indent}{target} := {value};\n"

    elif isinstance(stmt, ast.If):
        pseudo_code = f"{indent}IF ({translate_expression(stmt.test)}) THEN\n"
        for body_stmt in stmt.body:
            pseudo_code += translate_statement(body_stmt, indent_level + 1)
        if stmt.orelse:  # Handle 'else' blocks
            pseudo_code += f"{indent}ELSE\n"
            for orelse_stmt in stmt.orelse:
                pseudo_code += translate_statement(orelse_stmt, indent_level + 1)
        pseudo_code += f"{indent}END IF\n"
        return pseudo_code

    elif isinstance(stmt, ast.For):
        target = translate_target(stmt.target)
        iter_ = translate_expression(stmt.iter)
        pseudo_code = f"{indent}FOR EACH {target} IN {iter_}\n"
        for body_stmt in stmt.body:
            pseudo_code += translate_statement(body_stmt, indent_level + 1)
        pseudo_code += f"{indent}END FOR\n"
        return pseudo_code

    elif isinstance(stmt, ast.While):
        test = translate_expression(stmt.test)
        pseudo_code = f"{indent}WHILE ({test}) DO\n"
        for body_stmt in stmt.body:
            pseudo_code += translate_statement(body_stmt, indent_level + 1)
        pseudo_code += f"{indent}END WHILE\n"
        return pseudo_code

    elif isinstance(stmt, ast.Return):
        value = translate_expression(stmt.value) if stmt.value else ""
        return f"{indent}RETURN {value};\n"

    elif isinstance(stmt, ast.Expr):  # Handle function calls
        return f"{indent}{translate_expression(stmt.value)};\n"  # Just print the call

    elif isinstance(stmt, ast.Pass):
        return f"{indent}PASS;\n"

    # Add more statement types as needed (e.g., try-except, with, etc.)
    elif isinstance(stmt, ast.AnnAssign):  # Handle Type Hinting
        target = translate_target(stmt.target)
        annotation = translate_expression(stmt.annotation)
        value = translate_expression(stmt.value) if stmt.value else ""
        return f"{indent}{target} : {annotation} := {value};\n"
    elif isinstance(stmt, ast.AugAssign):  # Handle Augmented Assignments (+=, -=, etc.)
        target = translate_target(stmt.target)
        op = translate_operator(stmt.op)
        value = translate_expression(stmt.value)
        return f"{indent}{target} {op}= {value};\n"
    elif isinstance(stmt, ast.Raise):  # Handle Raise statements
        exc = translate_expression(stmt.exc) if stmt.exc else ""
        cause = translate_expression(stmt.cause) if stmt.cause else ""
        return f"{indent}RAISE {exc} FROM {cause};\n"
    elif isinstance(stmt, ast.Continue):
        return f"{indent}CONTINUE;\n"
    elif isinstance(stmt, ast.Break):
        return f"{indent}BREAK;\n"
    else:
        return f"{indent}# Unsupported statement type: {type(stmt)}\n"


def translate_target(target):
    if isinstance(target, ast.Name):
        return target.id
    elif isinstance(target, ast.Attribute):
        return f"{translate_target(target.value)}.{target.attr}"
    elif isinstance(target, ast.Subscript):
        return f"{translate_target(target.value)}[{translate_expression(target.slice)}]"
    # Add other target types if necessary
    return str(target)  # Default to string representation


def translate_expression(expr):
    if isinstance(expr, ast.Name):
        return expr.id
    elif isinstance(expr, ast.Constant):  # Handle constants (numbers, strings, booleans)
        return repr(expr.value)  # Use repr() for proper string representation
    elif isinstance(expr, ast.BinOp):
        left = translate_expression(expr.left)
        op = translate_operator(expr.op)
        right = translate_expression(expr.right)
        return f"{left} {op} {right}"
    elif isinstance(expr, ast.UnaryOp):
        op = translate_operator(expr.op)
        operand = translate_expression(expr.operand)
        return f"{op}{operand}"
    elif isinstance(expr, ast.Call):
        func = translate_expression(expr.func)
        args = ", ".join(translate_expression(arg) for arg in expr.args)
        keywords = ", ".join(f"{kw.arg}={translate_expression(kw.value)}" for kw in expr.keywords)
        if keywords:
            all_args = f"{args}, {keywords}" if args else keywords
        else:
            all_args = args
        return f"{func}({all_args})"
    elif isinstance(expr, ast.Compare):
        left = translate_expression(expr.left)
        ops = [translate_operator(op) for op in expr.ops]
        comparators = [translate_expression(comp) for comp in expr.comparators]
        comparison = f"{left} {ops[0]} {comparators[0]}"
        for i in range(1, len(ops)):
            comparison += f" {ops[i]} {comparators[i]}"
        return comparison
    elif isinstance(expr, ast.List):
        elements = ", ".join(translate_expression(elt) for elt in expr.elts)
        return f"[{elements}]"
    elif isinstance(expr, ast.Tuple):
        elements = ", ".join(translate_expression(elt) for elt in expr.elts)
        return f"({elements})"
    elif isinstance(expr, ast.Set):  # Handle sets
        elements = ", ".join(translate_expression(elt) for elt in expr.elts)
        return f"{{{elements}}}"
    elif isinstance(expr, ast.Dict):
        keys = ", ".join(translate_expression(key) for key in expr.keys)
        values = ", ".join(translate_expression(value) for value in expr.values)
        return f"{{{keys}: {values}}}"
    elif isinstance(expr, ast.Subscript):
        return f"{translate_expression(expr.value)}[{translate_expression(expr.slice)}]"
    elif isinstance(expr, ast.Attribute):
        return f"{translate_expression(expr.value)}.{expr.attr}"
    elif isinstance(expr, ast.Lambda):
        args = ", ".join(arg.arg for arg in expr.args.args)
        body = translate_expression(expr.body)
        return f"LAMBDA {args}: {body}"
    elif isinstance(expr, ast.IfExp):  # Ternary operator: x if condition else y
        test = translate_expression(expr.test)
        body = translate_expression(expr.body)
        orelse = translate_expression(expr.orelse)
        return f"{body} IF {test} ELSE {orelse}"
    elif isinstance(expr, ast.JoinedStr):  # f-strings
        values = "".join(translate_expression(v) for v in expr.values)
        return f'"{values}"'  # Enclose in double quotes
    elif isinstance(expr, ast.FormattedValue):  # Part of f-string
        value = translate_expression(expr.value)
        format_spec = translate_expression(expr.format_spec) if expr.format_spec else ""
        return f"{{{value}!{expr.conversion!s}{format_spec}}}"
    elif isinstance(expr, ast.Slice):  # Handle slice notation (e.g., [1:5], [:5], [1:])
        lower = translate_expression(expr.lower) if expr.lower is not None else ""
        upper = translate_expression(expr.upper) if expr.upper is not None else ""
        step = translate_expression(expr.step) if expr.step is not None else ""
        return f"{lower}:{upper}{':' + step if step else ''}"
    elif isinstance(expr, ast.ListComp):
        elt = translate_expression(expr.elt)
        generators = []
        for comp in expr.generators:
            target = translate_target(comp.target)
            iter_ = translate_expression(comp.iter)
            ifs = " ".join(f"IF {translate_expression(if_)}" for if_ in comp.ifs)
            generators.append(f"FOR {target} IN {iter_} {ifs}")
        return f"[{elt} {' '.join(generators)}]"
    elif isinstance(expr, ast.SetComp):
        elt = translate_expression(expr.elt)
        generators = []
        for comp in expr.generators:
            target = translate_target(comp.target)
            iter_ = translate_expression(comp.iter)
            ifs = " ".join(f"IF {translate_expression(if_)}" for if_ in comp.ifs)
            generators.append(f"FOR {target} IN {iter_} {ifs}")
        return f"{{{elt} {' '.join(generators)}}}"
    elif isinstance(expr, ast.DictComp):
        key = translate_expression(expr.key)
        value = translate_expression(expr.value)
        generators = []
        for comp in expr.generators:
            target = translate_target(comp.target)
            iter_ = translate_expression(comp.iter)
            ifs = " ".join(f"IF {translate_expression(if_)}" for if_ in comp.ifs)
            generators.append(f"FOR {target} IN {iter_} {ifs}")
        return f"{{{key}: {value} {' '.join(generators)}}}"

    else:
        return str(expr)  # Default to string representation


def translate_operator(op):
    if isinstance(op, ast.Add):
        return "+"
    elif isinstance(op, ast.Sub):
        return "-"
    elif isinstance(op, ast.Mult):
        return "*"
    elif isinstance(op, ast.Div):
        return "/"
    elif isinstance(op, ast.FloorDiv):
        return "//"
    elif isinstance(op, ast.Mod):
        return "%"
    elif isinstance(op, ast.Pow):
        return "**"
    elif isinstance(op, ast.LShift):
        return "<<"
    elif isinstance(op, ast.RShift):
        return ">>"
    elif isinstance(op, ast.BitOr):
        return "|"
    elif isinstance(op, ast.BitXor):
        return "^"
    elif isinstance(op, ast.BitAnd):
        return "&"
    elif isinstance(op, ast.MatMult):
        return "@"  # Matrix multiplication

    elif isinstance(op, ast.Eq):
        return "=="
    elif isinstance(op, ast.NotEq):
        return "!="
    elif isinstance(op, ast.Lt):
        return "<"
    elif isinstance(op, ast.Gt):
        return ">"
    elif isinstance(op, ast.LtE):
        return "<="
    elif isinstance(op, ast.GtE):
        return ">="
    elif isinstance(op, ast.Is):
        return "IS"
    elif isinstance(op, ast.IsNot):
        return "IS NOT"
    elif isinstance(op, ast.In):
        return "IN"
    elif isinstance(op, ast.NotIn):
        return "NOT IN"

    elif isinstance(op, ast.And):
        return "AND"
    elif isinstance(op, ast.Or):
        return "OR"
    elif isinstance(op, ast.Not):
        return "NOT"

    elif isinstance(op, ast.Invert):
        return "~"  # Bitwise not
    elif isinstance(op, ast.UAdd):
        return "+"  # Unary plus
    elif isinstance(op, ast.USub):
        return "-"  # Unary minus

    # Augmented assignment operators (e.g., +=, -=)
    elif isinstance(op, ast.AugAdd):
        return "+="
    elif isinstance(op, ast.AugSub):
        return "-="
    elif isinstance(op, ast.AugMult):
        return "*="
    elif isinstance(op, ast.AugDiv):
        return "/="
    elif isinstance(op, ast.AugFloorDiv):
        return "//="
    elif isinstance(op, ast.AugMod):
        return "%="
    elif isinstance(op, ast.AugPow):
        return "**="
    elif isinstance(op, ast.AugLShift):
        return "<<="
    elif isinstance(op, ast.AugRShift):
        return ">>="
    elif isinstance(op, ast.AugBitOr):
        return "|="
    elif isinstance(op, ast.AugBitXor):
        return "^="
    elif isinstance(op, ast.AugBitAnd):
        return "&="
    elif isinstance(op, ast.AugMatMult):
        return "@="

    else:
        return str(op)  # Default to string representation if not found


def main():
    python_file = 'pseudo.py'
    with open(python_file, 'r') as py_file_reader:
        src = py_file_reader.read()
        pseudocode = generate_pseudo_code(src)
        with open("%s_pseudo2.txt" % os.path.basename(python_file), "w") as wfp:
            wfp.write(pseudocode)
            wfp.write("\n")


if __name__ == '__main__':
    main()
