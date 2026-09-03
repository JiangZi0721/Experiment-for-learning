# etcd 的架构解密：Raft 共识与 bbolt MVCC 存储

## 1. 双层架构设计
etcd 分为两层：
- **共识层 (Raft)**：负责将写操作日志达成 Quorum 排序一致。
- **存储层 (MVCC + bbolt)**：内存中维护基于 B-Tree 的 `keyIndex`，磁盘底层采用基于 B+ Tree 的键值数据库 bbolt。

## 2. MVCC 修订版本号 (Revision)
- etcd 中每次事务修改，全局 `revision` 递增。
- 每个 key 的内部记录保留了 `create_revision`、`mod_revision` 和历史版本链。
- 这为 Kubernetes 提供了天然的增量变更监听（Watch）机制，客户端可以通过携带指定 revision 实现断点续传。
