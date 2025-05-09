use std::iter::Peekable;
use std::slice::Iter;
use crate::lexer::{Token, TokenType};
use crate::parser::ast::{ASTNode, Expression, Statement, Type, UnaryOperator};

pub fn parse(tokens: &Vec<Token>) -> Option<Vec<ASTNode>> {
    let mut iter = tokens.iter().peekable();
    let mut nodes = vec![];

    while let Some(token) = iter.peek() {
        match token.token_type {
            TokenType::Eof => break,
            TokenType::Let => {
                iter.next();
                if let Some(stmt) = parse_let(&mut iter) {
                    nodes.push(ASTNode::Statement(stmt));
                } else {
                    return None;
                }
            }
            TokenType::Print => {
                iter.next();
                if let Some(expr) = parse_expression(&mut iter) {
                    if matches!(iter.next(), Some(t) if t.token_type == TokenType::SemiColon) {
                        nodes.push(ASTNode::Statement(Statement::Print(expr)));
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            TokenType::Asm => {
                iter.next();
                if let Some(t) = iter.next() {
                    if t.token_type == TokenType::Lbrace {
                        let mut asm_code = String::new();
                        while let Some(tok) = iter.next() {
                            if tok.token_type == TokenType::Rbrace {
                                break;
                            }
                            asm_code += &tok.lexeme;
                            asm_code.push(' ');
                        }
                        nodes.push(ASTNode::Statement(Statement::InlineAsm(asm_code.trim().to_string())));
                    } else {
                        return None;
                    }
                }
            }
            _ => return None,
        }
    }
    Some(nodes)
}

fn parse_expression(tokens: &mut Peekable<Iter<'_, Token>>) -> Option<Expression> {
    use TokenType::*;

    let token = tokens.next()?;
    match &token.token_type {
        Int(value) => Some(Expression::Number(*value)),
        Identifier(s) => Some(Expression::Variable(s.clone())),
        Star => {
            let inner = parse_expression(tokens)?;
            Some(Expression::UnaryOp {
                op: UnaryOperator::Deref,
                expr: Box::new(inner),
            })
        }
        Ampersand => {
            let inner = parse_expression(tokens)?;
            Some(Expression::UnaryOp {
                op: UnaryOperator::AddressOf,
                expr: Box::new(inner),
            })
        }
        Minus => {
            let inner = parse_expression(tokens)?;
            Some(Expression::UnaryOp {
                op: UnaryOperator::Negate,
                expr: Box::new(inner),
            })
        }
        _ => None,
    }
}


pub fn parse_let(tokens: &mut Peekable<Iter<'_, Token>>) -> Option<Statement> {
    use TokenType::*;

    // 타입 파싱: int, char, void, int*
    let type_token = tokens.next()?;
    let mut base_type = match type_token.token_type {
        Int => Type::Int,
        Char => Type::Char,
        Void => Type::Void,
        _ => return None,
    };

    // 포인터인지 확인
    if let Some(next_token) = tokens.peek() {
        if next_token.token_type == Star {
            tokens.next();
            base_type = Type::Pointer(Box::new(base_type));
        }
    }

    // 변수 이름
    let name_token = tokens.next()?;
    let name = match &name_token.token_type {
        TokenType::Identifier(s) => s.clone(),
        _ => return None,
    };

    // =
    if tokens.next()?.token_type != TokenType::Equal {
        return None;
    }

    // 값 파싱
    let expr = parse_expression(tokens)?;

    // 세미콜론
    if tokens.next()?.token_type != TokenType::SemiColon {
        return None;
    }

    Some(Statement::Let {
        var_type: base_type,
        name,
        value: expr,
    })
}
