import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class LinkPredictionHead(nn.Module):
    """Task head for link prediction in knowledge graphs"""

    def __init__(self,
                 entity_dim: int,
                 relation_dim: int,
                 hidden_dim: int = 512,
                 num_relations: int = 50,
                 dropout: float = 0.1,
                 scoring_function: str = 'distmult',
                 use_bias: bool = True):
        super().__init__()

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.num_relations = num_relations
        self.scoring_function = scoring_function

        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)

        # Entity projection layers
        self.entity_proj_head = nn.Linear(entity_dim, hidden_dim)
        self.entity_proj_tail = nn.Linear(entity_dim, hidden_dim)

        # Scoring function specific components
        if scoring_function == 'distmult':
            self.relation_proj = nn.Linear(relation_dim, hidden_dim)
        elif scoring_function == 'complex':
            # ComplEx uses complex embeddings
            self.entity_proj_head_im = nn.Linear(entity_dim, hidden_dim)
            self.entity_proj_tail_im = nn.Linear(entity_dim, hidden_dim)
            self.relation_proj_im = nn.Linear(relation_dim, hidden_dim)
        elif scoring_function == 'rotate':
            # RotatE uses rotation in complex space
            self.relation_phase = nn.Embedding(num_relations, hidden_dim // 2)
        elif scoring_function == 'transe':
            # TransE uses translation
            self.margin = nn.Parameter(torch.tensor(1.0))

        # MLP for final scoring
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.entity_proj_head.weight)
        nn.init.xavier_uniform_(self.entity_proj_tail.weight)

        if self.scoring_function == 'rotate':
            nn.init.uniform_(self.relation_phase.weight, -torch.pi, torch.pi)

    def distmult_score(self, head: torch.Tensor, relation: torch.Tensor,
                      tail: torch.Tensor) -> torch.Tensor:
        """DistMult scoring function"""
        score = (head * relation * tail).sum(dim=-1)
        return score

    def complex_score(self, head: torch.Tensor, relation: torch.Tensor,
                     tail: torch.Tensor) -> torch.Tensor:
        """ComplEx scoring function"""
        # Split into real and imaginary parts
        head_re, head_im = torch.chunk(head, 2, dim=-1)
        rel_re, rel_im = torch.chunk(relation, 2, dim=-1)
        tail_re, tail_im = torch.chunk(tail, 2, dim=-1)

        # ComplEx score
        score = (head_re * rel_re * tail_re).sum(dim=-1) + \
                (head_re * rel_im * tail_im).sum(dim=-1) + \
                (head_im * rel_re * tail_im).sum(dim=-1) - \
                (head_im * rel_im * tail_re).sum(dim=-1)

        return score

    def rotate_score(self, head: torch.Tensor, relation: torch.Tensor,
                    tail: torch.Tensor) -> torch.Tensor:
        """RotatE scoring function"""
        pi = torch.pi
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # Get rotation phases
        phase_relation = relation / (self.entity_dim / 2 * pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # Rotate head
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        # Compute distance
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = -torch.norm(torch.stack([re_score, im_score], dim=0), dim=0, p=2).sum(dim=-1)

        return score

    def transe_score(self, head: torch.Tensor, relation: torch.Tensor,
                    tail: torch.Tensor) -> torch.Tensor:
        """TransE scoring function"""
        score = -torch.norm(head + relation - tail, p=2, dim=-1)
        return score

    def forward(self, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor,
                relation_ids: torch.Tensor, return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for link prediction

        Args:
            head_embeddings: Head entity embeddings [batch_size, entity_dim]
            tail_embeddings: Tail entity embeddings [batch_size, entity_dim]
            relation_ids: Relation IDs [batch_size]
            return_embeddings: Whether to return projected embeddings

        Returns:
            Dictionary containing scores and optionally embeddings
        """
        # Get relation embeddings
        relation_embeds = self.relation_embeddings(relation_ids)

        # Project entities
        head_proj = self.entity_proj_head(head_embeddings)
        tail_proj = self.entity_proj_tail(tail_embeddings)

        # Apply dropout
        head_proj = self.dropout(head_proj)
        tail_proj = self.dropout(tail_proj)

        # Score based on scoring function
        if self.scoring_function == 'distmult':
            relation_proj = self.relation_proj(relation_embeds)
            score = self.distmult_score(head_proj, relation_proj, tail_proj)
        elif self.scoring_function == 'complex':
            head_proj_im = self.entity_proj_head_im(head_embeddings)
            tail_proj_im = self.entity_proj_tail_im(tail_embeddings)
            relation_proj_im = self.relation_proj_im(relation_embeds)
            relation_proj = self.relation_proj(relation_embeds)

            head_complex = torch.cat([head_proj, head_proj_im], dim=-1)
            tail_complex = torch.cat([tail_proj, tail_proj_im], dim=-1)
            relation_complex = torch.cat([relation_proj, relation_proj_im], dim=-1)

            score = self.complex_score(head_complex, relation_complex, tail_complex)
        elif self.scoring_function == 'rotate':
            phase = self.relation_phase(relation_ids)
            score = self.rotate_score(head_proj, phase, tail_proj)
        elif self.scoring_function == 'transe':
            relation_proj = self.relation_proj(relation_embeds)
            score = self.transe_score(head_proj, relation_proj, tail_proj)
        else:
            # Default: use MLP scoring
            combined = head_proj + relation_embeds + tail_proj
            score = self.score_mlp(combined).squeeze(-1)

        # Add bias
        if self.bias is not None:
            score = score + self.bias

        output = {
            'scores': score,
            'probabilities': torch.sigmoid(score)
        }

        if return_embeddings:
            output['head_embeddings'] = head_proj
            output['tail_embeddings'] = tail_proj
            output['relation_embeddings'] = relation_embeds

        return output

    def compute_loss(self, scores: torch.Tensor, labels: torch.Tensor,
                    negative_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute link prediction loss"""
        if negative_scores is not None:
            # Margin-based loss
            positive_scores = scores[labels == 1]
            negative_scores = scores[labels == 0]

            if self.scoring_function == 'transe':
                loss = F.relu(self.margin - positive_scores + negative_scores).mean()
            else:
                loss = -F.logsigmoid(positive_scores).mean() - \
                       F.logsigmoid(-negative_scores).mean()
        else:
            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(scores, labels)

        return loss