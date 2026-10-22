import torch


class MaterialFrontier:
    """Material frontier targets aligned to a supplied material-name order."""

    def __init__(self, hierarchy, material_names):
        names = [self._norm(x) for x in material_names]
        index = {name: i for i, name in enumerate(names)}
        parent = {
            self._norm(child): (None if par is None else self._norm(par))
            for child, par in hierarchy['parent'].items()
        }

        k = len(names)
        strict_descendant = torch.zeros((k, k), dtype=torch.bool)
        for descendant in names:
            current = parent.get(descendant)
            while current is not None:
                strict_descendant[index[current], index[descendant]] = True
                current = parent.get(current)

        self.strict_descendant = strict_descendant

    @staticmethod
    def _norm(value):
        return str(value).strip().lower()

    def frontier(self, target):
        """Keep positive nodes with no positive descendant in the same row."""
        positive = target > 0
        descendants = self.strict_descendant.to(target.device)
        has_positive_descendant = (positive.float() @ descendants.float().T) > 0
        frontier = positive & ~has_positive_descendant

        valid = positive.any(dim=1)
        empty = valid & ~frontier.any(dim=1)
        if empty.any():
            frontier[empty] = positive[empty]
        return frontier.to(target.dtype)
