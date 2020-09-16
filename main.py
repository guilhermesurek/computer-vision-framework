from opts.opts import Options

def pretty_print(title, msg, size=24):
    ''' Make print prettier. '''
    print(size * "-")
    print(("{:^" + str(size) + "}").format(title))
    print(size * "-")
    print(f"{msg}\n")

if __name__ == "__main__":
    # Get options - Inspired by Kensho Hara
    opt = Options().opts

    # Print loaded options
    pretty_print("OPTIONS", opt)