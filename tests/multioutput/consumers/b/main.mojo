from moxie import ConsumerA, ConsumerB

def main() raises:
    ConsumerA().run()
    ConsumerB().run()
    print("PASS: isolated shared consumer")
