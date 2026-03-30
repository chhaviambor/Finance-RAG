import asyncio
from fastapi import FastAPI
import app

async def run_tests():
    fake_app = FastAPI()
    print("Spinning up RAG for Tests...")
    
    async with app.lifespan(fake_app):
        if app.qa_chain is None:
            print("FAILED: qa_chain failed to initialize.")
            return

        print("\n====== TEST 1: IN-DOMAIN ======")
        q1 = "What did the committee decide regarding the federal funds rate?"
        print("Q:", q1)
        res1 = app.qa_chain.invoke(q1)
        print("A:", res1)
        
        print("\n====== TEST 2: OUT-OF-DOMAIN ======")
        q2 = "How do you bake a chocolate cake?"
        print("Q:", q2)
        res2 = app.qa_chain.invoke(q2)
        print("A:", res2)
        
        print("\n====== TEST 3: CONTEXT MISSING ======")
        q3 = "What were the exact profitability margins of Google in 2024?"
        print("Q:", q3)
        res3 = app.qa_chain.invoke(q3)
        print("A:", res3)

if __name__ == "__main__":
    asyncio.run(run_tests())
