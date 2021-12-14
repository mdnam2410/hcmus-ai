import copy
from functools import total_ordering
import itertools
from typing import Union

@total_ordering
class Literal:
    def __init__(self, string):
        sym = string.strip('-')
        if len(sym) != 1:
            raise ValueError
        self.symbol = sym
        self.neg = string.startswith('-')

    def is_negative(self):
        return self.neg

    def complement(self, other):
        return self.symbol == other.symbol and self.neg != other.neg

    def __str__(self):
        return ('-' if self.neg else '') + self.symbol

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.symbol == other.symbol and self.neg == other.neg

    def __lt__(self, other):
        if self.symbol == other.symbol:
            # A is smaller than not(A)
            return int(self.neg) < int(other.neg)
        return self.symbol < other.symbol

    def __hash__(self) -> int:
        return hash((self.symbol, self.neg))


class Clause:
    """Represents a disjunction of a list of literals

    Examples:
        * `Clause([Literal('U')])` represents the clause with a single
        literal U
        * `Clause([Literal('A'), Literal('-B')])` represents A ∨ ¬B
        * `Clause.fromstring('P OR -Q OR R')` represents P ∨ ¬Q ∨ R

    A list of Clause represents the conjuction of the clauses. 
        * `[Clause(Literal('A')), Clause(Literal('B'))]` represents A ∧ ¬B
        * `[Clause.fromstring('P OR Q'), Clause.fromstring('-R')]`
        represents (P ∨ Q) ∧ R
    """
    def __init__(self, list_of_literals):
        self.literals = list_of_literals
        self._factor()

    @classmethod
    def fromstring(cls, string):
        lits = [] if string == '' else string.split(' OR ')
        lits = [Literal(lit) for lit in lits]
        return cls(lits)

    def is_empty(self):
        return len(self.literals) == 0

    def is_equivalent_to_true(self):
        """Returns True if the clause contains a pair of complement literals
        """
        i = 0
        while i < len(self) - 1:
            if self.literals[i].complement(self.literals[i + 1]):
                return True
            i += 1
        return False
    
    def symbols(self):
        return set(x.symbol for x in self.literals)

    def _factor(self):
        """Removes multiple copies of literals and sorts in alphabetical
        order
        """
        self.literals = list(set(self.literals))
        self.literals.sort()

    def __str__(self):
        if self.is_empty():
            return r'{}'
        return ' OR '.join([str(lit) for lit in self.literals])
    
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.literals == other.literals
    
    def __hash__(self):
        return hash(tuple(self.literals))

    def __len__(self):
        return len(self.literals)


def negate(term: Union[Literal, Clause]):
    """Negates a literal or a clause

    Args:
        term (Literal): A Literal or
        term (Clause): A Clause

    Returns:
        Literal: if `type(term) == Literal`
        list[Clause]: if `type(term) = Clause`
    """
    if type(term) == Literal:
        return Literal(('' if term.is_negative() else '-') + term.symbol)
    return [Clause([negate(lit)]) for lit in term.literals]
    
def resolve(clause1, clause2):
    """Resolves two clauses

    Returns:
        Clause: A new clause
        None: If two clauses don't resolve to anything
    """
    # Construct a list of literals from the two clauses
    lits = clause1.literals + clause2.literals
    lits = sorted(list(set(lits)))
    
    # Pop the first found pair of complement literals
    i = 0
    has_complement_pair = False
    while i < len(lits) - 1:
        if lits[i].complement(lits[i + 1]):
            has_complement_pair = True
            lits.pop(i)
            lits.pop(i)
            break
        i += 1
    
    # Two clauses are resolvable if one complement pair is found
    return Clause(lits) if has_complement_pair else None


def pl_resolution(kb, alpha, result_buffer) -> bool:
    """Determines if `kb` entails `alpha`

    Args:
        kb (list): A list of clause
        alpha (Clause): A single clause
        result_buffer (list): Buffer into which the result of each iteration
        is written

    Returns:
        bool: Returns True if `kb` entails `alpha`
    """
    clauses = kb + negate(alpha)
    new = []

    i = 0
    while True:
        new_resolvents = []
        explanations = []
        for x, y in itertools.product(clauses, repeat=2):
            resolvent = resolve(x, y)

            # Not resolvable
            if resolvent is None:
                continue

            # Resolvent is chosen if it has not appeared anywhere and is not
            # equivalent to True (i.e. contains a pair of complement literals)
            if resolvent not in new and \
               resolvent not in new_resolvents and \
               resolvent not in clauses and \
               not resolvent.is_equivalent_to_true():
                new_resolvents.append(resolvent)
        
        result_buffer.append((len(new_resolvents), copy.copy(new_resolvents)))

        # `kb` entails `alpha` if the empty clause is resolved
        if Clause([]) in new_resolvents:
            result_buffer.append('YES')
            return True
        
        new.extend(new_resolvents)

        # `kb` does not entail `alpha` if no new resolvent is made
        if set(new).issubset(set(clauses)):
            result_buffer.append('NO')
            return False
        
        clauses = list(set(clauses).union(new))
        i += 1


def run(test_number):
    with open(f'input/input{test_number}.txt') as input:
        alpha = input.readline().strip()
        num_clauses = int(input.readline().strip())

        clauses = []
        for _ in range(num_clauses):
            clauses.append(input.readline().strip())

        kb = [Clause.fromstring(x) for x in clauses]
        alpha = Clause.fromstring(alpha)
        write_buffer = []
        
        print(f'Test {test_number} result: {pl_resolution(kb, alpha, write_buffer)}')

    with open(f'output/output{test_number}.txt', mode='w+') as output:
        for elem in write_buffer:
            if type(elem) == tuple:
                num_resolvents, resolvents = elem
                output.write(str(num_resolvents) + '\n')
                output.writelines([str(x) + '\n' for x in resolvents])
            else:
                output.writelines([elem])


if __name__ == '__main__':
    num_tests = 5
    for i in range(num_tests):
        run(i + 1)
