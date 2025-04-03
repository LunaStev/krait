#!/usr/bin/env python3
import sys

# --- 토큰 타입 정의 ---
TT_INT         = 'INT'
TT_IDENT       = 'IDENT'
TT_STRING      = 'STRING'
TT_KEYWORD     = 'KEYWORD'
TT_PLUS        = 'PLUS'
TT_MINUS       = 'MINUS'
TT_MUL         = 'MUL'
TT_DIV         = 'DIV'
TT_LPAREN      = 'LPAREN'
TT_RPAREN      = 'RPAREN'
TT_LBRACE      = 'LBRACE'
TT_RBRACE      = 'RBRACE'
TT_ASSIGN      = 'ASSIGN'
TT_SEMI        = 'SEMI'
TT_COLON       = 'COLON'
TT_COMMA       = 'COMMA'
TT_EOF         = 'EOF'
TT_DIRECTIVE   = 'DIRECTIVE'   # 예: #define ...
TT_INLINE_ASM  = 'INLINE_ASM'  # asm { ... }
TT_AMPERSAND   = 'AMPERSAND'   # & (주소 연산자)
TT_LBRACKET    = 'LBRACKET'    # [
TT_RBRACKET    = 'RBRACKET'    # ]
TT_CHAR        = 'CHAR'        # 문자 리터럴: 'A'

# 지원하는 키워드 (추가: int, void, char, asm, namespace, import, typedef 등)
KEYWORDS = [
    'let', 'print', 'if', 'elif', 'else', 'while', 'for', 'in',
    'return', 'case', 'when', 'default', 'range', 'def', 'fn', 'typedef',
    'int', 'void', 'char', 'asm', 'namespace', 'import'
]

# --- 토큰 클래스 ---
class Token:
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value
    def __repr__(self):
        return f"{self.type}:{self.value}" if self.value is not None else self.type

# --- 렉서 ---
class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = text[self.pos] if text else None

    def advance(self):
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        return None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def read_string(self):
        # 문자열 리터럴: "..."
        self.advance()  # 시작 " 소비
        result = ''
        while self.current_char is not None and self.current_char != '"':
            result += self.current_char
            self.advance()
        if self.current_char != '"':
            raise Exception("Unterminated string literal")
        self.advance()  # 종료 " 소비
        return result

    def read_char(self):
        # 문자 리터럴: 'A'
        self.advance()  # 시작 ' 소비
        if self.current_char is None:
            raise Exception("Unterminated character literal")
        char = self.current_char
        self.advance()
        if self.current_char != "'":
            raise Exception("Unterminated character literal")
        self.advance()  # 종료 ' 소비
        return char

    def integer(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def identifier(self):
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        if result in KEYWORDS:
            # 인라인 어셈블리: asm { ... }
            if result == "asm":
                self.skip_whitespace()
                if self.current_char == '{':
                    self.advance()  # '{' 소비
                    asm_code = self.read_until_rbrace()
                    return Token(TT_INLINE_ASM, asm_code)
            return Token(TT_KEYWORD, result)
        return Token(TT_IDENT, result)

    def read_until_rbrace(self):
        result = ''
        while self.current_char is not None and self.current_char != '}':
            result += self.current_char
            self.advance()
        if self.current_char != '}':
            raise Exception("Unterminated inline assembly block")
        self.advance()  # '}' 소비
        return result.strip()

    def read_directive(self):
        result = ''
        while self.current_char is not None and self.current_char != '\n':
            result += self.current_char
            self.advance()
        return result.strip()

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char == '#':
                directive = self.read_directive()
                return Token(TT_DIRECTIVE, directive)
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            if self.current_char == '"':
                return Token(TT_STRING, self.read_string())
            if self.current_char == "'":
                return Token(TT_CHAR, self.read_char())
            if self.current_char.isdigit():
                return Token(TT_INT, self.integer())
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()
            if self.current_char == '+':
                self.advance()
                return Token(TT_PLUS)
            if self.current_char == '-':
                self.advance()
                return Token(TT_MINUS)
            if self.current_char == '*':
                self.advance()
                return Token(TT_MUL)
            if self.current_char == '/':
                self.advance()
                return Token(TT_DIV)
            if self.current_char == '(':
                self.advance()
                return Token(TT_LPAREN)
            if self.current_char == ')':
                self.advance()
                return Token(TT_RPAREN)
            if self.current_char == '{':
                self.advance()
                return Token(TT_LBRACE)
            if self.current_char == '}':
                self.advance()
                return Token(TT_RBRACE)
            if self.current_char == '[':
                self.advance()
                return Token(TT_LBRACKET)
            if self.current_char == ']':
                self.advance()
                return Token(TT_RBRACKET)
            if self.current_char == '&':
                self.advance()
                return Token(TT_AMPERSAND)
            if self.current_char == '=':
                self.advance()
                return Token(TT_ASSIGN)
            if self.current_char == ';':
                self.advance()
                return Token(TT_SEMI)
            if self.current_char == ':':
                self.advance()
                return Token(TT_COLON)
            if self.current_char == ',':
                self.advance()
                return Token(TT_COMMA)
            raise Exception(f"Illegal character: {self.current_char}")
        return Token(TT_EOF)

# --- AST 노드 ---
class AST: pass

class Number(AST):
    def __init__(self, value):
        self.value = value

class CharLiteral(AST):
    def __init__(self, value):
        # 문자 리터럴은 내부적으로 정수(아스키 코드)로 저장
        self.value = ord(value)

class Var(AST):
    def __init__(self, name):
        self.name = name

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op  # TT_PLUS, TT_MINUS 등
        self.right = right

# 새로운 노드: 단항 연산 (포인터 역참조, 주소 연산 등)
class UnaryOp(AST):
    def __init__(self, op, operand):
        self.op = op  # TT_PLUS, TT_MINUS, TT_MUL, TT_AMPERSAND
        self.operand = operand

# 새로운 노드: 배열 인덱싱 (x[i])
class IndexAccess(AST):
    def __init__(self, var, index_expr):
        self.var = var
        self.index_expr = index_expr

# let 문 – 선택적 타입(var_type) 지원
class LetStatement(AST):
    def __init__(self, var_name, expr, var_type=None):
        self.var_name = var_name
        self.expr = expr
        self.var_type = var_type

class PrintStatement(AST):
    def __init__(self, expr):
        self.expr = expr

class ReturnStatement(AST):
    def __init__(self, expr):
        self.expr = expr

class IfStatement(AST):
    def __init__(self, condition, then_block, elif_clauses=None, else_block=None):
        self.condition = condition
        self.then_block = then_block
        self.elif_clauses = elif_clauses if elif_clauses is not None else []
        self.else_block = else_block

class WhileStatement(AST):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ForStatement(AST):
    def __init__(self, var, iterable, body):
        self.var = var
        self.iterable = iterable
        self.body = body

class FunctionCall(AST):
    def __init__(self, func, args):
        self.func = func
        self.args = args

class CaseClause(AST):
    def __init__(self, value, block):
        self.value = value
        self.block = block

class CaseStatement(AST):
    def __init__(self, expr, clauses, default_clause=None):
        self.expr = expr
        self.clauses = clauses
        self.default_clause = default_clause

class FunctionDefinition(AST):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

# typedef – 필드는 (타입, 필드명) 튜플 리스트
class TypeDef(AST):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields

# 매크로 정의 (#define ...)
class MacroDefinition(AST):
    def __init__(self, name, value):
        self.name = name
        self.value = value

# 인라인 어셈블리 (asm { ... })
class InlineAssembly(AST):
    def __init__(self, asm_code):
        self.asm_code = asm_code

# namespace: namespace IDENT { statements }
class Namespace(AST):
    def __init__(self, name, body):
        self.name = name
        self.body = body

# import: import "filename.krait";
class ImportStatement(AST):
    def __init__(self, filename, ast_nodes):
        self.filename = filename
        self.ast_nodes = ast_nodes

# --- 파서 ---
class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            raise Exception(f"Unexpected token: {self.current_token}, expected {token_type}")

    # 새로 추가: postfix 연산 (함수 호출, 배열 인덱싱)
    def postfix(self, node):
        while self.current_token.type in (TT_LPAREN, TT_LBRACKET):
            if self.current_token.type == TT_LPAREN:
                node = self.function_call(node)
            elif self.current_token.type == TT_LBRACKET:
                node = self.index_access(node)
        return node

    def index_access(self, node):
        self.eat(TT_LBRACKET)
        index_expr = self.expr()
        self.eat(TT_RBRACKET)
        return IndexAccess(node, index_expr)

    # factor: 단항 연산자 포함 및 postfix 처리
    def factor(self):
        token = self.current_token
        if token.type in (TT_PLUS, TT_MINUS, TT_MUL, TT_AMPERSAND):
            self.eat(token.type)
            operand = self.factor()
            return UnaryOp(token.type, operand)
        elif token.type == TT_INT:
            self.eat(TT_INT)
            return Number(token.value)
        elif token.type == TT_CHAR:
            self.eat(TT_CHAR)
            return CharLiteral(token.value)
        elif token.type == TT_IDENT:
            self.eat(TT_IDENT)
            node = Var(token.value)
            node = self.postfix(node)
            return node
        elif token.type == TT_LPAREN:
            self.eat(TT_LPAREN)
            node = self.expr()
            self.eat(TT_RPAREN)
            node = self.postfix(node)
            return node
        else:
            raise Exception("Invalid syntax in factor")

    def term(self):
        node = self.factor()
        while self.current_token.type in (TT_MUL, TT_DIV):
            # 주의: 앞의 TT_MUL이 단항 연산자로 처리되었으므로 여기서의 TT_MUL은 이항 곱셈
            op = self.current_token.type
            if op == TT_MUL:
                self.eat(TT_MUL)
            elif op == TT_DIV:
                self.eat(TT_DIV)
            node = BinOp(node, op, self.factor())
        return node

    def expr(self):
        node = self.term()
        while self.current_token.type in (TT_PLUS, TT_MINUS):
            op = self.current_token.type
            if op == TT_PLUS:
                self.eat(TT_PLUS)
            elif op == TT_MINUS:
                self.eat(TT_MINUS)
            node = BinOp(node, op, self.term())
        return node

    # let 문: let [타입]? IDENT '=' expr ';'
    def parse_let_statement(self):
        self.eat(TT_KEYWORD)  # let
        var_type = None
        if self.current_token.type == TT_KEYWORD and self.current_token.value in ("int", "void", "char"):
            var_type = self.current_token.value
            self.eat(TT_KEYWORD)
        if self.current_token.type != TT_IDENT:
            raise Exception("Expected identifier after let")
        var_name = self.current_token.value
        self.eat(TT_IDENT)
        self.eat(TT_ASSIGN)
        expr = self.expr()
        self.eat(TT_SEMI)
        return LetStatement(var_name, expr, var_type)

    # typedef 내 필드: let [타입]? IDENT ';'
    def parse_typedef_field(self):
        self.eat(TT_KEYWORD)  # let
        field_type = None
        if self.current_token.type == TT_KEYWORD and self.current_token.value in ("int", "void", "char"):
            field_type = self.current_token.value
            self.eat(TT_KEYWORD)
        if self.current_token.type != TT_IDENT:
            raise Exception("Expected field name after let in typedef")
        field_name = self.current_token.value
        self.eat(TT_IDENT)
        self.eat(TT_SEMI)
        return (field_type, field_name)

    # namespace: namespace IDENT { statement* }
    def parse_namespace(self):
        self.eat(TT_KEYWORD)  # namespace
        if self.current_token.type != TT_IDENT:
            raise Exception("Expected namespace name")
        ns_name = self.current_token.value
        self.eat(TT_IDENT)
        body = self.parse_block()
        return Namespace(ns_name, body)

    # import: import "filename.krait";
    def parse_import_statement(self):
        self.eat(TT_KEYWORD)  # import
        if self.current_token.type != TT_STRING:
            raise Exception("Expected string literal for import")
        filename = self.current_token.value
        self.eat(TT_STRING)
        self.eat(TT_SEMI)
        # 파일 읽기 및 파싱 (실패 시 예외 발생)
        try:
            with open(filename, "r") as f:
                imported_source = f.read()
        except Exception as e:
            raise Exception(f"Failed to import file {filename}: {e}")
        imported_lexer = Lexer(imported_source)
        imported_parser = Parser(imported_lexer)
        imported_ast = imported_parser.parse()
        return ImportStatement(filename, imported_ast)

    # 블록: { statement* }
    def parse_block(self):
        self.eat(TT_LBRACE)
        statements = []
        while self.current_token.type != TT_RBRACE:
            statements.append(self.statement())
        self.eat(TT_RBRACE)
        return statements

    # 함수 호출: IDENT '(' (expr (',' expr)*)? ')'
    def function_call(self, func_node):
        self.eat(TT_LPAREN)
        args = []
        if self.current_token.type != TT_RPAREN:
            args.append(self.expr())
            while self.current_token.type == TT_COMMA:
                self.eat(TT_COMMA)
                args.append(self.expr())
        self.eat(TT_RPAREN)
        return FunctionCall(func_node, args)

    def statement(self):
        if self.current_token.type == TT_DIRECTIVE:
            return self.parse_directive()
        if self.current_token.type == TT_INLINE_ASM:
            asm_token = self.current_token
            self.eat(TT_INLINE_ASM)
            return InlineAssembly(asm_token.value)
        if self.current_token.type == TT_KEYWORD:
            kw = self.current_token.value
            if kw == 'namespace':
                return self.parse_namespace()
            elif kw == 'import':
                return self.parse_import_statement()
            elif kw == 'let':
                return self.parse_let_statement()
            elif kw == 'print':
                self.eat(TT_KEYWORD)
                node = self.expr()
                self.eat(TT_SEMI)
                return PrintStatement(node)
            elif kw == 'return':
                self.eat(TT_KEYWORD)
                node = self.expr()
                self.eat(TT_SEMI)
                return ReturnStatement(node)
            elif kw == 'if':
                return self.parse_if_statement()
            elif kw == 'while':
                return self.parse_while_statement()
            elif kw == 'for':
                return self.parse_for_statement()
            elif kw == 'case':
                return self.parse_case_statement()
            elif kw in ('def', 'fn'):
                return self.parse_function_definition()
            elif kw == 'typedef':
                return self.parse_typedef()
            else:
                raise Exception(f"Unknown keyword in statement: {kw}")
        else:
            raise Exception("Unknown statement")

    def parse_directive(self):
        directive_text = self.current_token.value  # 예: "#define MAX 100"
        self.eat(TT_DIRECTIVE)
        parts = directive_text.split()
        if len(parts) < 3 or parts[0] != "#define":
            raise Exception("Invalid directive format")
        macro_name = parts[1]
        macro_value = " ".join(parts[2:])
        return MacroDefinition(macro_name, macro_value)

    def parse_if_statement(self):
        self.eat(TT_KEYWORD)  # if
        condition = self.expr()
        then_block = self.parse_block()
        elif_clauses = []
        while self.current_token.type == TT_KEYWORD and self.current_token.value == 'elif':
            self.eat(TT_KEYWORD)
            cond = self.expr()
            block = self.parse_block()
            elif_clauses.append((cond, block))
        else_block = None
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'else':
            self.eat(TT_KEYWORD)
            else_block = self.parse_block()
        return IfStatement(condition, then_block, elif_clauses, else_block)

    def parse_while_statement(self):
        self.eat(TT_KEYWORD)  # while
        condition = self.expr()
        body = self.parse_block()
        return WhileStatement(condition, body)

    def parse_for_statement(self):
        self.eat(TT_KEYWORD)  # for
        if self.current_token.type != TT_IDENT:
            raise Exception("Expected identifier in for loop")
        var_name = self.current_token.value
        self.eat(TT_IDENT)
        if not (self.current_token.type == TT_KEYWORD and self.current_token.value == 'in'):
            raise Exception("Expected 'in' in for loop")
        self.eat(TT_KEYWORD)
        iterable = self.expr()
        body = self.parse_block()
        return ForStatement(var_name, iterable, body)

    def parse_case_statement(self):
        self.eat(TT_KEYWORD)  # case
        case_expr = self.expr()
        clauses, default_clause = self.parse_case_block()
        return CaseStatement(case_expr, clauses, default_clause)

    def parse_case_block(self):
        self.eat(TT_LBRACE)
        clauses = []
        default_clause = None
        while self.current_token.type != TT_RBRACE:
            if self.current_token.type == TT_KEYWORD and self.current_token.value == 'when':
                self.eat(TT_KEYWORD)
                value_expr = self.expr()
                self.eat(TT_COLON)
                stmts = []
                while (self.current_token.type != TT_KEYWORD or 
                       (self.current_token.value not in ('when','default'))) and self.current_token.type != TT_RBRACE:
                    stmts.append(self.statement())
                clauses.append(CaseClause(value_expr, stmts))
            elif self.current_token.type == TT_KEYWORD and self.current_token.value == 'default':
                self.eat(TT_KEYWORD)
                self.eat(TT_COLON)
                stmts = []
                while self.current_token.type != TT_RBRACE:
                    stmts.append(self.statement())
                default_clause = stmts
            else:
                raise Exception("Invalid token in case block")
        self.eat(TT_RBRACE)
        return clauses, default_clause

    def parse_function_definition(self):
        kw = self.current_token.value
        self.eat(TT_KEYWORD)  # def 또는 fn
        if self.current_token.type != TT_IDENT:
            raise Exception("Expected function name after def/fn")
        func_name = self.current_token.value
        self.eat(TT_IDENT)
        self.eat(TT_LPAREN)
        params = []
        if self.current_token.type != TT_RPAREN:
            if self.current_token.type != TT_IDENT:
                raise Exception("Expected parameter name")
            params.append(self.current_token.value)
            self.eat(TT_IDENT)
            while self.current_token.type == TT_COMMA:
                self.eat(TT_COMMA)
                if self.current_token.type != TT_IDENT:
                    raise Exception("Expected parameter name")
                params.append(self.current_token.value)
                self.eat(TT_IDENT)
        self.eat(TT_RPAREN)
        body = self.parse_block()
        return FunctionDefinition(func_name, params, body)

    def parse_typedef(self):
        self.eat(TT_KEYWORD)  # typedef
        if self.current_token.type != TT_IDENT:
            raise Exception("Expected type name after typedef")
        type_name = self.current_token.value
        self.eat(TT_IDENT)
        self.eat(TT_LBRACE)
        fields = []
        while self.current_token.type != TT_RBRACE:
            if self.current_token.type == TT_KEYWORD and self.current_token.value == 'let':
                fields.append(self.parse_typedef_field())
            else:
                raise Exception("Expected field declaration in typedef")
        self.eat(TT_RBRACE)
        if self.current_token.type == TT_SEMI:
            self.eat(TT_SEMI)
        return TypeDef(type_name, fields)

    def parse(self):
        statements = []
        while self.current_token.type != TT_EOF:
            statements.append(self.statement())
        return statements

# --- 코드 생성기 ---
class CodeGenerator:
    def __init__(self, ast):
        self.ast = ast
        self.global_statements = []
        self.macros = []
        # namespace는 전역 문장으로 처리 (나머지는 모두 global_statements)
        for node in ast:
            if isinstance(node, MacroDefinition):
                self.macros.append(node)
            else:
                self.global_statements.append(node)
        self.code = []
        self.label_count = 0
        self.exit_label = self.new_label()
        self.namespace_prefix = ""  # 전역에서는 접두사 없음
        self.variables = set()
        self.collect_variables(self.global_statements)

    def new_label(self):
        label = f"L{self.label_count}"
        self.label_count += 1
        return label

    # 전역 변수(let 문) 수집 (namespace 내부는 compile_statement에서 처리)
    def collect_variables(self, statements):
        for stmt in statements:
            if isinstance(stmt, LetStatement):
                self.variables.add(stmt.var_name)
            elif isinstance(stmt, ForStatement):
                self.variables.add(stmt.var)
                self.collect_variables(stmt.body)
            elif isinstance(stmt, IfStatement):
                self.collect_variables(stmt.then_block)
                for (_, block) in stmt.elif_clauses:
                    self.collect_variables(block)
                if stmt.else_block:
                    self.collect_variables(stmt.else_block)
            elif isinstance(stmt, WhileStatement):
                self.collect_variables(stmt.body)
            elif isinstance(stmt, CaseStatement):
                for clause in stmt.clauses:
                    self.collect_variables(clause.block)
                if stmt.default_clause:
                    self.collect_variables(stmt.default_clause)
            elif isinstance(stmt, Namespace):
                self.collect_variables(stmt.body)

    def generate(self):
        # 매크로 정의 출력 (NASM의 %define)
        for macro in self.macros:
            self.code.append(f"%define {macro.name} {macro.value}")
        # typedef – 전역에 주석으로 출력
        for stmt in self.global_statements:
            if isinstance(stmt, TypeDef):
                self.code.append(f"; typedef {stmt.name}:")
                for i, (f_type, field) in enumerate(stmt.fields):
                    type_str = f_type + " " if f_type else ""
                    self.code.append(f";    {type_str}{field}: offset {i*8}")
        # 전역 변수 (.bss) – namespace 접두사는 compile 시점에 적용됨
        self.code.append("extern printf")
        self.code.append("section .data")
        self.code.append('    fmt: db "%d", 10, 0')
        self.code.append("section .bss")
        for var in self.variables:
            self.code.append(f"    {self.namespace_prefix}{var}: resq 1")
        self.code.append("section .text")
        self.code.append("global main")
        self.code.append("main:")
        # 전역 실행 코드 (global_statements 중 let, print, return, inline asm, namespace, import 등)
        for stmt in self.global_statements:
            self.code.extend(self.compile_statement(stmt))
        self.code.append(f"{self.exit_label}:")
        self.code.append("    mov rax, 60")
        self.code.append("    xor rdi, rdi")
        self.code.append("    syscall")
        return "\n".join(self.code)

    def compile_statement(self, stmt, context=None):
        if isinstance(stmt, LetStatement):
            code = []
            code.extend(self.compile_expr(stmt.expr, context))
            if stmt.var_type:
                comment = f"; let {stmt.var_type} {stmt.var_name}"
            else:
                comment = f"; let {stmt.var_name}"
            code.insert(0, comment)
            if context is not None:
                if stmt.var_name not in context["locals"]:
                    context["current_offset"] += 8
                    context["locals"][stmt.var_name] = context["current_offset"]
                offset = context["locals"][stmt.var_name]
                code.append(f"    mov [rbp - {offset}], rax")
            else:
                code.append(f"    mov [ {self.namespace_prefix}{stmt.var_name} ], rax")
            return code
        elif isinstance(stmt, PrintStatement):
            code = []
            code.extend(self.compile_expr(stmt.expr, context))
            code.append("    mov rsi, rax")
            code.append("    lea rdi, [rel fmt]")
            code.append("    xor rax, rax")
            code.append("    call printf")
            return code
        elif isinstance(stmt, ReturnStatement):
            code = []
            code.extend(self.compile_expr(stmt.expr, context))
            code.append("    pop rbp")
            code.append("    ret")
            return code
        elif isinstance(stmt, IfStatement):
            return self.compile_if(stmt, context)
        elif isinstance(stmt, WhileStatement):
            return self.compile_while(stmt, context)
        elif isinstance(stmt, ForStatement):
            return self.compile_for(stmt, context)
        elif isinstance(stmt, CaseStatement):
            return self.compile_case(stmt, context)
        elif isinstance(stmt, FunctionDefinition):
            return self.compile_function(stmt)
        elif isinstance(stmt, InlineAssembly):
            return [stmt.asm_code]
        elif isinstance(stmt, TypeDef):
            return []  # typedef는 이미 주석으로 출력됨
        elif isinstance(stmt, MacroDefinition):
            return []  # 매크로는 이미 출력됨
        elif isinstance(stmt, Namespace):
            old_prefix = self.namespace_prefix
            self.namespace_prefix = old_prefix + stmt.name + "_"
            code = []
            self.collect_variables(stmt.body)
            for s in stmt.body:
                code.extend(self.compile_statement(s))
            self.namespace_prefix = old_prefix
            return code
        elif isinstance(stmt, ImportStatement):
            code = []
            for s in stmt.ast_nodes:
                code.extend(self.compile_statement(s))
            return code
        else:
            raise Exception("Unknown statement in code generation")

    def compile_expr(self, node, context=None):
        if isinstance(node, Number):
            return [f"    mov rax, {node.value}"]
        elif isinstance(node, CharLiteral):
            return [f"    mov rax, {node.value}"]
        elif isinstance(node, Var):
            if context is not None:
                if node.name in context["params"]:
                    return [f"    mov rax, {context['params'][node.name]}"]
                elif node.name in context["locals"]:
                    offset = context["locals"][node.name]
                    return [f"    mov rax, QWORD [rbp - {offset}]"]
                else:
                    return [f"    mov rax, [ {self.namespace_prefix}{node.name} ]"]
            else:
                return [f"    mov rax, [ {self.namespace_prefix}{node.name} ]"]
        elif isinstance(node, UnaryOp):
            if node.op == TT_PLUS:
                return self.compile_expr(node.operand, context)
            elif node.op == TT_MINUS:
                code = []
                code.extend(self.compile_expr(node.operand, context))
                code.append("    mov rcx, rax")
                code.append("    mov rax, 0")
                code.append("    sub rax, rcx")
                return code
            elif node.op == TT_MUL:  # 포인터 역참조
                code = []
                code.extend(self.compile_expr(node.operand, context))
                code.append("    mov rax, [rax]")
                return code
            elif node.op == TT_AMPERSAND:  # 주소 연산자
                if isinstance(node.operand, Var):
                    return [f"    lea rax, [ {self.namespace_prefix}{node.operand.name} ]"]
                else:
                    raise Exception("Address-of operator only supported for variables")
        elif isinstance(node, BinOp):
            if node.op == TT_PLUS:
                code = []
                code.extend(self.compile_expr(node.left, context))
                code.append("    push rax")
                code.extend(self.compile_expr(node.right, context))
                code.append("    pop rbx")
                code.append("    add rax, rbx")
                return code
            elif node.op == TT_MINUS:
                code = []
                code.extend(self.compile_expr(node.left, context))
                code.append("    push rax")
                code.extend(self.compile_expr(node.right, context))
                code.append("    mov rcx, rax")
                code.append("    pop rbx")
                code.append("    mov rax, rbx")
                code.append("    sub rax, rcx")
                return code
            elif node.op == TT_MUL:
                code = []
                code.extend(self.compile_expr(node.left, context))
                code.append("    push rax")
                code.extend(self.compile_expr(node.right, context))
                code.append("    pop rbx")
                code.append("    imul rax, rbx")
                return code
            elif node.op == TT_DIV:
                code = []
                code.extend(self.compile_expr(node.left, context))
                code.append("    push rax")
                code.extend(self.compile_expr(node.right, context))
                code.append("    mov rcx, rax")
                code.append("    pop rbx")
                code.append("    mov rax, rbx")
                code.append("    cqo")
                code.append("    idiv rcx")
                return code
        elif isinstance(node, FunctionCall):
            code = []
            param_order = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
            for i, arg in enumerate(node.args):
                code.extend(self.compile_expr(arg, context))
                if i < len(param_order):
                    code.append(f"    mov {param_order[i]}, rax")
                else:
                    raise Exception("More than 6 arguments not supported")
            if isinstance(node.func, Var):
                code.append(f"    call {self.namespace_prefix}{node.func.name}")
                return code
            else:
                raise Exception("Function call must be a variable (function name)")
        elif isinstance(node, IndexAccess):
            code = []
            # 기본 포인터 주소 획득
            code.extend(self.compile_expr(node.var, context))
            code.append("    push rax")
            # 인덱스 계산
            code.extend(self.compile_expr(node.index_expr, context))
            code.append("    pop rbx")
            code.append("    add rax, rbx")
            # 메모리에서 바이트 읽기 (문자열 등은 1바이트 단위)
            code.append("    movzx rax, byte [rax]")
            return code
        else:
            raise Exception("Unknown expression node")

    def compile_if(self, node, context):
        code = []
        end_label = self.new_label()
        code.extend(self.compile_expr(node.condition, context))
        false_label = self.new_label()
        code.append("    cmp rax, 0")
        code.append(f"    je {false_label}")
        for stmt in node.then_block:
            code.extend(self.compile_statement(stmt, context))
        code.append(f"    jmp {end_label}")
        code.append(f"{false_label}:")
        for (cond, block) in node.elif_clauses:
            next_label = self.new_label()
            code.extend(self.compile_expr(cond, context))
            code.append("    cmp rax, 0")
            code.append(f"    je {next_label}")
            for stmt in block:
                code.extend(self.compile_statement(stmt, context))
            code.append(f"    jmp {end_label}")
            code.append(f"{next_label}:")
        if node.else_block:
            for stmt in node.else_block:
                code.extend(self.compile_statement(stmt, context))
        code.append(f"{end_label}:")
        return code

    def compile_while(self, node, context):
        code = []
        start_label = self.new_label()
        end_label = self.new_label()
        code.append(f"{start_label}:")
        code.extend(self.compile_expr(node.condition, context))
        code.append("    cmp rax, 0")
        code.append(f"    je {end_label}")
        for stmt in node.body:
            code.extend(self.compile_statement(stmt, context))
        code.append(f"    jmp {start_label}")
        code.append(f"{end_label}:")
        return code

    def compile_for(self, node, context):
        code = []
        if not (isinstance(node.iterable, FunctionCall) and isinstance(node.iterable.func, Var) and node.iterable.func.name == "range"):
            raise Exception("For loop iterable must be a range() call")
        if len(node.iterable.args) != 2:
            raise Exception("range() must have two arguments: start and end")
        code.extend(self.compile_expr(node.iterable.args[0], context))
        if context is not None:
            if node.var not in context["locals"]:
                context["current_offset"] += 8
                context["locals"][node.var] = context["current_offset"]
            offset = context["locals"][node.var]
            code.append(f"    mov [rbp - {offset}], rax")
        else:
            code.append(f"    mov [ {self.namespace_prefix}{node.var} ], rax")
        code.extend(self.compile_expr(node.iterable.args[1], context))
        code.append("    mov rbx, rax")
        start_label = self.new_label()
        end_label = self.new_label()
        code.append(f"{start_label}:")
        if context is not None:
            offset = context["locals"][node.var]
            code.append(f"    mov rax, QWORD [rbp - {offset}]")
        else:
            code.append(f"    mov rax, [ {self.namespace_prefix}{node.var} ]")
        code.append("    cmp rax, rbx")
        code.append(f"    jge {end_label}")
        for stmt in node.body:
            code.extend(self.compile_statement(stmt, context))
        if context is not None:
            offset = context["locals"][node.var]
            code.append(f"    mov rax, QWORD [rbp - {offset}]")
            code.append("    add rax, 1")
            code.append(f"    mov [rbp - {offset}], rax")
        else:
            code.append(f"    mov rax, [ {self.namespace_prefix}{node.var} ]")
            code.append("    add rax, 1")
            code.append(f"    mov [ {self.namespace_prefix}{node.var} ], rax")
        code.append(f"    jmp {start_label}")
        code.append(f"{end_label}:")
        return code

    def compile_case(self, node, context):
        code = []
        code.extend(self.compile_expr(node.expr, context))
        end_label = self.new_label()
        clause_labels = []
        for clause in node.clauses:
            label = self.new_label()
            clause_labels.append(label)
            if not isinstance(clause.value, Number):
                raise Exception("Case clause value must be a constant number")
            code.append(f"    cmp rax, {clause.value.value}")
            code.append(f"    je {label}")
        if node.default_clause:
            default_label = self.new_label()
            code.append(f"    jmp {default_label}")
        else:
            code.append(f"    jmp {end_label}")
        for i, clause in enumerate(node.clauses):
            code.append(f"{clause_labels[i]}:")
            for stmt in clause.block:
                code.extend(self.compile_statement(stmt, context))
            code.append(f"    jmp {end_label}")
        if node.default_clause:
            code.append(f"{default_label}:")
            for stmt in node.default_clause:
                code.extend(self.compile_statement(stmt, context))
            code.append(f"    jmp {end_label}")
        code.append(f"{end_label}:")
        return code

    def compile_function(self, func_node):
        code = []
        func_name = self.namespace_prefix + func_node.name
        code.append(f"{func_name}:")
        code.append("    push rbp")
        code.append("    mov rbp, rsp")
        param_registers = {}
        param_order = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        for i, param in enumerate(func_node.params):
            if i < len(param_order):
                param_registers[param] = param_order[i]
            else:
                raise Exception("More than 6 parameters not supported")
        local_context = {
            "params": param_registers,
            "locals": {},
            "current_offset": 0
        }
        for stmt in func_node.body:
            code.extend(self.compile_statement(stmt, local_context))
        code.append("    mov rax, 0")
        code.append("    pop rbp")
        code.append("    ret")
        return code

def main():
    if len(sys.argv) < 2:
        print("Usage: python krait_transpiler.py <input_file.krait>")
        sys.exit(1)
    input_file = sys.argv[1]
    if not input_file.endswith(".krait"):
        print("Input file must have .krait extension")
        sys.exit(1)
    with open(input_file, "r") as f:
        source = f.read()
    lexer = Lexer(source)
    parser = Parser(lexer)
    ast = parser.parse()
    generator = CodeGenerator(ast)
    asm_code = generator.generate()
    output_file = input_file.rsplit(".", 1)[0] + ".asm"
    with open(output_file, "w") as f:
        f.write(asm_code)
    print(f"Transpilation complete. Output written to {output_file}")

if __name__ == "__main__":
    main()
