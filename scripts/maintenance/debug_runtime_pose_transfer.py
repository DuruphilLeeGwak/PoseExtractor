import inspect
import os
import sys


def main() -> None:
    # Ensure workspace root is importable regardless of how Python is invoked.
    workspace_root = os.path.abspath(os.path.dirname(__file__))
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

    print("cwd:", os.getcwd())
    print("sys.executable:", sys.executable)
    print("sys.path[0:5]:")
    for p in sys.path[:5]:
        print("  ", p)

    import pose_transfer
    import pose_transfer.transfer.engine as eng
    import pose_transfer.transfer.logic.body as body

    print("pose_transfer.__file__:", pose_transfer.__file__)
    print("engine.__file__:", eng.__file__)
    print("body.__file__:", body.__file__)

    print(
        "BodyTransfer.transfer_shoulders signature:",
        inspect.signature(body.BodyTransfer.transfer_shoulders),
    )

    print("engine has _calculate_torso_ratio:", hasattr(eng, "_calculate_torso_ratio"))
    if hasattr(eng, "_calculate_torso_ratio"):
        print("_calculate_torso_ratio signature:", inspect.signature(eng._calculate_torso_ratio))

    src = inspect.getsource(eng.PoseTransferEngine.transfer)
    print("\nLines in PoseTransferEngine.transfer containing transfer_shoulders:")
    for line in src.splitlines():
        if "transfer_shoulders" in line:
            print("  " + line.strip())


if __name__ == "__main__":
    main()
