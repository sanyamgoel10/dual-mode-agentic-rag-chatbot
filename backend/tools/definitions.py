TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Search company documents using semantic search. Use for questions about "
            "return policy, warranty, HR leave policy, product FAQ, pricing and discounts. "
            "Returns relevant text with source document name and page number for citation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_orders",
        "description": (
            "Query the orders database. Use for questions about orders, revenue, customers, "
            "products sold, order counts, or order status. "
            "Write a SELECT SQL query. Table name: orders. "
            "Columns: order_id (TEXT), customer (TEXT), product (TEXT), "
            "amount (REAL, in rupees), status (TEXT: pending/shipped/delivered/cancelled), "
            "order_date (TEXT: YYYY-MM-DD format)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SELECT SQL query against the orders table",
                }
            },
            "required": ["sql"],
        },
    },
]
