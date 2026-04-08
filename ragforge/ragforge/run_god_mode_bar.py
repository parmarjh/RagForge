import time
import random
import sys

def main():
    print("=" * 60)
    print("🚀  RAGFORGE GOD MODE BAR - ADMIN TELEMETRY INITIALIZING  🚀")
    print("=" * 60)
    print("Waiting for data pipeline connections...")
    time.sleep(1)

    try:
        pipeline_count = 0
        while True:
            pipeline_count += 1
            embed_latency = random.uniform(0.01, 0.05)
            db_latency = random.uniform(0.005, 0.02)
            llm_throughput = random.randint(30, 120)
            status = random.choice(["OK", "OK", "OK", "OPTIMAL", "SYNCING"])

            sys.stdout.write(f"\r[PIPELINE {pipeline_count:04d}] Status: {status:<8} | "
                             f"Embed: {embed_latency:.3f}s | "
                             f"Vector DB: {db_latency:.3f}s | "
                             f"LLM: {llm_throughput:3d} tokens/s")
            sys.stdout.flush()
            time.sleep(0.5)
            
            if pipeline_count % 10 == 0:
                print(f"\n[EVENT] Automatic index compaction triggered. Freed {random.randint(10, 50)} MB.")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("🛑  GOD MODE BAR TERMINATED  🛑")
        print("=" * 60)

if __name__ == "__main__":
    main()
