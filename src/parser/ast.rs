#[derive(Debug, Clone)]
pub enum ASTNode {
    Statement(Statement),
    Expression(Expression),
}

pub struct Program {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Let {
        var_type: Type,
        name: String,
        value: Expression,
    },
    Print(Expression),
    InlineAsm(String),
}

#[derive(Clone, Debug)]
pub enum Type {
    Int,
    Char,
    Void,
    Pointer(Box<Type>),
}

#[derive(Clone, Debug)]
pub enum Expression {
    Number(i64),
    Variable(String),
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expression>,
    },
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOperator,
        right: Box<Expression>,
    },
}

#[derive(Clone, Debug)]
pub enum UnaryOperator {
    Negate,     // -
    AddressOf,  // &
    Deref,      // *
}

#[derive(Clone, Debug)]
pub enum BinaryOperator {
    Add,    // +
    Sub,    // -
    Mul,    // *
    Div,    // /
}
