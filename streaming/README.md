\# Kafka Streaming



This directory contains the Kafka-based transaction streaming layer for the

Real-Time Fraud Detection \& Risk Scoring Platform.



\## Architecture



```text

IEEE-CIS Validation Dataset

&#x20;         |

&#x20;         v

&#x20;  Kafka Producer

&#x20;         |

&#x20;         v

&#x20;fraud-transactions topic

&#x20;         |

&#x20;   +-----+-----+

&#x20;   |     |     |

&#x20;   v     v     v

&#x20; P0      P1    P2

&#x20;         |

&#x20;         v

&#x20;  Kafka Consumer

&#x20;         |

&#x20;         v

&#x20;Future Fraud Detection Pipeline

