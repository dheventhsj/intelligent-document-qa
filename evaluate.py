"""Run retrieval comparisons; optionally export generated answers for human review."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from core import build_index, retrieve, answer


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--docs', default='sample_docs')
    parser.add_argument('--questions', default='eval_questions.json')
    args = parser.parse_args()
    files = [(p.name, p.read_bytes()) for p in sorted(Path(args.docs).iterdir()) if p.suffix.lower() in {'.txt', '.md', '.pdf', '.docx'}]
    questions = json.loads(Path(args.questions).read_text())
    rows = []
    for size, overlap in [(256, 32), (384, 64), (512, 64)]:
        store, count = build_index(files, size, overlap)
        for k in [2, 4]:
            cases = []
            for item in questions:
                hits = retrieve(store, item['question'], k)
                expected = item['source']
                rank = next((i for i, (d, _) in enumerate(hits, 1) if d.metadata['source'] == expected and item['evidence'].lower() in d.page_content.lower()), None) if expected else None
                case = {**item, 'evidence_hit': bool(rank), 'reciprocal_rank': 1 / rank if rank else 0,
                        'retrieval_rejected': not hits,
                        'retrieved': [{'source': d.metadata['source'], 'page': d.metadata['page'], 'distance': s, 'text': d.page_content} for d, s in hits]}
                if args.generate:
                    case['answer'], case['citation_status'] = answer(item['question'], hits)
                cases.append(case)
            answerable = [c for c in cases if c['source']]
            if not answerable:
                raise ValueError('Evaluation requires at least one answerable question.')
            row = {'size': size, 'overlap': overlap, 'k': k, 'chunks': count,
                   'evidence_hit_at_k': sum(c['evidence_hit'] for c in answerable) / len(answerable),
                   'MRR_at_k': sum(c['reciprocal_rank'] for c in answerable) / len(answerable), 'cases': cases}
            rows.append(row)
            print({key: val for key, val in row.items() if key != 'cases'})
    Path('results').mkdir(exist_ok=True)
    Path('results/evaluation.json').write_text(json.dumps(rows, indent=2))
    print('Saved results/evaluation.json. Review correctness and grounding manually; retrieval metrics are not answer quality metrics.')


if __name__ == '__main__':
    main()
