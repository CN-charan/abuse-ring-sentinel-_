"""
Layer 1: Data Generation for Abuse-Ring Sentinel
--------------------------------------------------
Produces:
  accounts.csv       -> one row per account, with identifiers
  ground_truth.csv   -> account_id, is_fraud_ring (1/0), ring_id
Design choices (from our architecture):
  - 95% normal accounts, 5% in abuse rings
  - Normal accounts sometimes share LOW-risk signals (IP, address) via
    "family/household" groups of 2-5 people -> this is the noise that
    makes the problem hard, and it should NEVER cause a false flag.
  - Ring accounts share HIGH-risk signals (device_id, card) but only
    1-2 of them, staggered across ring members (not everyone shares
    with everyone) -> this is what makes rings hard to catch with a
    naive rule.
"""

import random
import uuid
import pandas as pd

random.seed(42)  # reproducibility matters for a judge re-running your code

# ---- CONFIG: change these to tune scale ----
N_TOTAL_ACCOUNTS = 2000
RING_FRACTION = 0.05          # ~5% of accounts belong to a ring
RING_SIZE_RANGE = (5, 10)     # accounts per ring
HOUSEHOLD_SIZE_RANGE = (2, 5) # innocent IP/address sharing group size
HOUSEHOLD_FRACTION = 0.15     # 15% of normal accounts belong to a household


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def make_account(account_id):
    """A fully independent, unique account -- no sharing yet."""
    return {
        "account_id": account_id,
        "device_id": new_id("dev"),
        "ip": new_id("ip"),
        "card": new_id("card"),
        "address": new_id("addr"),
    }


def generate_normal_accounts(n):
    """Generate n normal accounts, with some innocent household sharing."""
    accounts = [make_account(new_id("acct")) for _ in range(n)]

    n_household_accounts = int(n * HOUSEHOLD_FRACTION)
    i = 0
    while i < n_household_accounts:
        size = random.randint(*HOUSEHOLD_SIZE_RANGE)
        group = accounts[i:i + size]
        if len(group) < 2:
            break
        # Household shares ONE of {ip, address} -- never device or card.
        # This is the "innocent noise" that should not trigger a flag.
        shared_signal = random.choice(["ip", "address"])
        shared_value = group[0][shared_signal]
        for acct in group:
            acct[shared_signal] = shared_value
        i += size

    return accounts


def generate_ring_accounts(n):
    """Generate accounts belonging to abuse rings.
    Rings share 1-2 HIGH-risk signals (device_id, card), staggered --
    not every member shares with every other member, which is what
    makes plain connected_components potentially fragment the ring.
    """
    accounts = []
    ring_id_counter = 0
    remaining = n

    while remaining > 0:
        size = min(random.randint(*RING_SIZE_RANGE), remaining)
        ring_id = f"ring_{ring_id_counter}"
        ring_id_counter += 1

        ring_members = [make_account(new_id("acct")) for _ in range(size)]

        # Decide which high-risk signal(s) this ring shares: device, card, or both
        signals_to_share = random.sample(["device_id", "card"], k=random.choice([1, 2]))

        for signal in signals_to_share:
            shared_value = new_id(signal[:3])
            # STAGGERED sharing: only a random subset (60-90%) of members
            # get the shared value, not all of them. This is what makes
            # it a graph-clustering problem, not a simple groupby.
            n_sharing = max(2, int(size * random.uniform(0.6, 0.9)))
            sharers = random.sample(ring_members, n_sharing)
            for acct in sharers:
                acct[signal] = shared_value

        for acct in ring_members:
            acct["ring_id"] = ring_id

        accounts.extend(ring_members)
        remaining -= size

    return accounts


def main():
    n_ring_accounts = int(N_TOTAL_ACCOUNTS * RING_FRACTION)
    n_normal_accounts = N_TOTAL_ACCOUNTS - n_ring_accounts

    normal = generate_normal_accounts(n_normal_accounts)
    rings = generate_ring_accounts(n_ring_accounts)

    all_accounts = normal + rings
    random.shuffle(all_accounts)  # don't leak order as a signal

    accounts_df = pd.DataFrame(all_accounts)

    ground_truth = accounts_df[["account_id"]].copy()
    ground_truth["is_fraud_ring"] = accounts_df.get("ring_id").notna().astype(int)
    ground_truth["ring_id"] = accounts_df.get("ring_id")

    accounts_df = accounts_df.drop(columns=["ring_id"], errors="ignore")

    accounts_df.to_csv("accounts.csv", index=False)
    ground_truth.to_csv("ground_truth.csv", index=False)

    print(f"Generated {len(accounts_df)} accounts")
    print(f"  Normal: {len(normal)} ({HOUSEHOLD_FRACTION*100:.0f}% in households)")
    print(f"  Ring members: {len(rings)} across {ground_truth['ring_id'].nunique()} rings")
    print("Wrote accounts.csv and ground_truth.csv")


if __name__ == "__main__":
    main()
